//includes
#include "pose_estimation.hpp"
#include "glogger.hpp"
//TODO
//add file names
//add reference to logger

bool pose_estimation::compileOnnxToEngine(const std::string& onnxPath, const std::string& enginePath, cv::Size targetSize) {
	std::cout << "\n========================================================" << std::endl;
	std::cout << "TensorRT 10 Engine Compiler Active Engine Optimization" << std::endl;
	std::cout << "Building from: " << onnxPath << std::endl;
	std::cout << "========================================================\n" << std::endl;

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	if (!builder) return false;

	// Use strongly typed network configurations (TensorRT 10 native pattern)
	uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flags);
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

	if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
		std::cerr << "Critical Error: Failed to parse structural ONNX configuration layers." << std::endl;
		delete parser;
		delete network; 
		delete builder;
        return false;
    }

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    
	int numInputs = network->getNbInputs();
	bool hasDynamic = false;

	for (int i = 0; i < numInputs; ++i) {
		nvinfer1::ITensor* input = network->getInput(i);
		nvinfer1::Dims dims = input->getDimensions();
		const char* inputName = input->getName();
        
		bool isInputDynamic = false;
			
		for (int d = 0; d < dims.nbDims; ++d) {
			if (dims.d[d] == -1) {
				isInputDynamic = true;
				hasDynamic = true;
            }
        }

        if (isInputDynamic) {
            nvinfer1::Dims minDims = dims;
            nvinfer1::Dims optDims = dims;
            nvinfer1::Dims maxDims = dims;

            if (minDims.d[0] == -1) {
                minDims.d[0] = 1; optDims.d[0] = 1; maxDims.d[0] = 1;
            }

            if (std::string(inputName) == "input0" || std::string(inputName) == "input_1:0") {
                if (dims.nbDims == 4) {
                    minDims.d[2] = targetSize.height; minDims.d[3] = targetSize.width;
                    optDims.d[2] = targetSize.height; optDims.d[3] = targetSize.width;
                    maxDims.d[2] = targetSize.height; maxDims.d[3] = targetSize.width;
                }
            } else {
                for (int d = 1; d < dims.nbDims; ++d) {
                    if (dims.d[d] == -1) {
                        minDims.d[d] = 3; optDims.d[d] = 3; maxDims.d[d] = 3;
                    }
                }
            }

            profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims);
            profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims);
            profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);
            std::cout << "Added optimization profile mapping for dynamic input: " << inputName << std::endl;
        }
    }

    // Assign the profile if it was requested. If not needed, builder ownership handles cleanup.
    if (hasDynamic) {
        config->addOptimizationProfile(profile);
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

	// Let the builder auto-optimize target arrays internally via strongly typed layout constraints
    std::cout << "Hardware mapping validation initialized..." << std::endl;

    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (!serializedModel) {
        std::cerr << "Critical Error: Model optimization engine generation failed." << std::endl;
        delete config;
		delete parser;
		delete network;
		delete builder;
        return false;
    }

    std::ofstream engineFile(enginePath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    engineFile.close();

    std::cout << "\n>>> Production Engine Compiled and Saved to Disk: " << enginePath << " <<<\n" << std::endl;

    delete serializedModel;
	delete config;
	delete parser;
	delete network;
	delete builder;
    return true;
}

bool pose_estimation::initializeBodyPose3D(const std::string& engine_file) {
    std::vector<char> engine_data = loadEngineFile(engine_file);
    if (engine_data.empty()) return false;

    bp_ctx.runtime.reset(nvinfer1::createInferRuntime(gLogger));
    bp_ctx.engine.reset(bp_ctx.runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    bp_ctx.context.reset(bp_ctx.engine->createExecutionContext());

    cudaStreamCreate(&bp_ctx.stream);

    // 1. Allocate Unified Memory directly to the pointers
    cudaMallocManaged((void**)&bp_ctx.d_input0, 1 * 3 * bp_config.input_h * bp_config.input_w * sizeof(float));
    cudaMallocManaged((void**)&bp_ctx.d_k_inv, 1 * 3 * 3 * sizeof(float));
    cudaMallocManaged((void**)&bp_ctx.d_t_form_inv, 1 * 3 * 3 * sizeof(float));
    cudaMallocManaged((void**)&bp_ctx.d_scale_norm_limb, 1 * 36 * sizeof(float));
    cudaMallocManaged((void**)&bp_ctx.d_mean_limb, 1 * 36 * sizeof(float));

    cudaMallocManaged((void**)&bp_ctx.d_pose2d, 1 * bp_config.num_keypoints * 3 * sizeof(float));
    cudaMallocManaged((void**)&bp_ctx.d_pose2d_org, 1 * bp_config.num_keypoints * 3 * sizeof(float));
    cudaMallocManaged((void**)&bp_ctx.d_pose25d, 1 * bp_config.num_keypoints * 4 * sizeof(float));
    cudaMallocManaged((void**)&bp_ctx.d_pose3d, 1 * bp_config.num_keypoints * 3 * sizeof(float));

    // (Host vector .resize() calls removed from here)

    // 2. Set Tensor Addresses
    bp_ctx.context->setTensorAddress("input0", bp_ctx.d_input0);
    bp_ctx.context->setTensorAddress("k_inv", bp_ctx.d_k_inv);
    bp_ctx.context->setTensorAddress("t_form_inv", bp_ctx.d_t_form_inv);
    bp_ctx.context->setTensorAddress("scale_normalized_mean_limb_lengths", bp_ctx.d_scale_norm_limb);
    bp_ctx.context->setTensorAddress("mean_limb_lengths", bp_ctx.d_mean_limb);

    bp_ctx.context->setTensorAddress("pose2d", bp_ctx.d_pose2d);
    bp_ctx.context->setTensorAddress("pose2d_org_img", bp_ctx.d_pose2d_org);
    bp_ctx.context->setTensorAddress("pose25d", bp_ctx.d_pose25d);
    bp_ctx.context->setTensorAddress("pose3d", bp_ctx.d_pose3d);

    cv::Mat k_inv_float;
    this->geo.cameraMatrixInverse.convertTo(k_inv_float, CV_32F);
    k_inv_float = k_inv_float.clone();

    // 3. Use standard std::memcpy to populate static Unified Memory (No cudaMemcpyAsync needed)
    std::memcpy(bp_ctx.d_k_inv, k_inv_float.ptr<float>(), 9 * sizeof(float));
    std::memcpy(bp_ctx.d_scale_norm_limb, bp_config.scale_normalized_mean_limb_lengths.data(), 36 * sizeof(float));
    std::memcpy(bp_ctx.d_mean_limb, bp_config.mean_limb_lengths.data(), 36 * sizeof(float));

    return true;
}
		
std::vector<char> pose_estimation::loadEngineFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.good()) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}
		
void pose_estimation::processAndRunBodyPose(const cv::Mat& blob, const cv::Mat& t_form_inv) {
    // 1. Write dynamic frame data directly to unified memory
    std::memcpy(bp_ctx.d_input0, blob.ptr<float>(), blob.total() * sizeof(float));
    std::memcpy(bp_ctx.d_t_form_inv, t_form_inv.ptr<float>(), 9 * sizeof(float));

    // 2. Run Inference
    bp_ctx.context->enqueueV3(bp_ctx.stream);

    // 3. Wait for GPU to finish! (No DeviceToHost copies needed)
    cudaStreamSynchronize(bp_ctx.stream);
}

void pose_estimation::cleanupBodyPose3D(BodyPoseContext& trt) {
    if (trt.stream) cudaStreamDestroy(trt.stream);
    if (trt.d_input0) cudaFree(trt.d_input0);
    if (trt.d_k_inv) cudaFree(trt.d_k_inv);
    if (trt.d_t_form_inv) cudaFree(trt.d_t_form_inv);
    if (trt.d_scale_norm_limb) cudaFree(trt.d_scale_norm_limb);
    if (trt.d_mean_limb) cudaFree(trt.d_mean_limb);
    if (trt.d_pose2d) cudaFree(trt.d_pose2d);
    if (trt.d_pose2d_org) cudaFree(trt.d_pose2d_org);
    if (trt.d_pose25d) cudaFree(trt.d_pose25d);
    if (trt.d_pose3d) cudaFree(trt.d_pose3d);
}

int pose_estimation::setup(std::string engine_file, std::string onnx_file, cv::Size targetSize, CameraGeometry& loaded_geo, bool rebuild){
	//load calibration data needed
	//handle engine creation
	//compileOnnxToEngine(onnx_file, engine_file, targetSize);
	//initializeBodyPose3D(engine_file);
		
	this->geo = loaded_geo;
		
	 if (access(engine_file.c_str(), F_OK) == -1 || rebuild) {
		std::cout << "Notice: Compiled execution target file '" << engine_file << "' not found." << std::endl;
        if (!compileOnnxToEngine(onnx_file, engine_file, targetSize)) {
			std::cout << "compileing body pose engine file failed" << std::endl;
			return -1;
		}
    }

	if (!initializeBodyPose3D(engine_file)) {
        std::cerr << "Failed to initialize BodyPose3D engine." << std::endl;
        return -1;
    }

	std::cout << "body pose runner setup succesfull" << std::endl;
	return 1;
}

//add reference to output data 
int pose_estimation::run(const cv::Mat& blob, const cv::Mat& t_form_inv){
	processAndRunBodyPose(blob, t_form_inv);
	return 1;
}

BodyPoseContext* pose_estimation::getContextPtr() { 
        return &bp_ctx; 
}

BodyPoseConfig* pose_estimation::getConfigPtr(){ 
	return &bp_config; 
}
		//constructor
pose_estimation::pose_estimation(CameraGeometry in_geo) {
	geo = in_geo;
}
	
		//destructor
pose_estimation::~pose_estimation(){
	;		
}
