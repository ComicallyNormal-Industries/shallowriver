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

    cudaMalloc(&bp_ctx.d_input0, 1 * 3 * bp_config.input_h * bp_config.input_w * sizeof(float));
    cudaMalloc(&bp_ctx.d_k_inv, 1 * 3 * 3 * sizeof(float));
    cudaMalloc(&bp_ctx.d_t_form_inv, 1 * 3 * 3 * sizeof(float));
    cudaMalloc(&bp_ctx.d_scale_norm_limb, 1 * 36 * sizeof(float));
    cudaMalloc(&bp_ctx.d_mean_limb, 1 * 36 * sizeof(float));

    cudaMalloc(&bp_ctx.d_pose2d, 1 * bp_config.num_keypoints * 3 * sizeof(float));
    cudaMalloc(&bp_ctx.d_pose2d_org, 1 * bp_config.num_keypoints * 3 * sizeof(float));
   	cudaMalloc(&bp_ctx.d_pose25d, 1 * bp_config.num_keypoints * 4 * sizeof(float));
    cudaMalloc(&bp_ctx.d_pose3d, 1 * bp_config.num_keypoints * 3 * sizeof(float));

    bp_ctx.h_pose3d.resize(bp_config.num_keypoints * 3);
    bp_ctx.h_pose2d_org.resize(bp_config.num_keypoints * 3);
    bp_ctx.h_pose2d.resize(bp_config.num_keypoints * 3);
    bp_ctx.h_pose25d.resize(bp_config.num_keypoints * 4);

    bp_ctx.context->setTensorAddress("input0", bp_ctx.d_input0);
    bp_ctx.context->setTensorAddress("k_inv", bp_ctx.d_k_inv);
   	bp_ctx.context->setTensorAddress("t_form_inv", bp_ctx.d_t_form_inv);
    bp_ctx.context->setTensorAddress("scale_normalized_mean_limb_lengths", bp_ctx.d_scale_norm_limb);
    bp_ctx.context->setTensorAddress("mean_limb_lengths", bp_ctx.d_mean_limb);
    
    bp_ctx.context->setTensorAddress("pose2d", bp_ctx.d_pose2d);
    bp_ctx.context->setTensorAddress("pose2d_org_img", bp_ctx.d_pose2d_org);
    bp_ctx.context->setTensorAddress("pose25d", bp_ctx.d_pose25d);
    bp_ctx.context->setTensorAddress("pose3d", bp_ctx.d_pose3d);

    cudaMemcpyAsync(bp_ctx.d_scale_norm_limb, bp_config.scale_normalized_mean_limb_lengths.data(), 36 * sizeof(float), cudaMemcpyHostToDevice, bp_ctx.stream);
    cudaMemcpyAsync(bp_ctx.d_mean_limb, bp_config.mean_limb_lengths.data(), 36 * sizeof(float), cudaMemcpyHostToDevice, bp_ctx.stream);
   	cudaStreamSynchronize(bp_ctx.stream);

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
		
void pose_estimation::processAndRunBodyPose(const cv::Mat& original_frame, const cv::Rect& person_box) {
    
    cv::Point2f src_pts[3], dst_pts[3];
    src_pts[0] = cv::Point2f(person_box.x, person_box.y);
    src_pts[1] = cv::Point2f(person_box.x + person_box.width, person_box.y);
    src_pts[2] = cv::Point2f(person_box.x, person_box.y + person_box.height);
    
    dst_pts[0] = cv::Point2f(0, 0);
    dst_pts[1] = cv::Point2f(bp_config.input_w, 0);
    dst_pts[2] = cv::Point2f(0, bp_config.input_h);

    //t_form_inv allocation and type precision ---
    cv::Mat t_form = cv::getAffineTransform(src_pts, dst_pts);
    cv::Mat t_form_3x3 = cv::Mat::eye(3, 3, CV_32F);
    t_form.convertTo(t_form_3x3(cv::Rect(0, 0, 3, 2)), CV_32F);
    
    // Assign to a temporary cv::Mat first to resolve the expression
    cv::Mat t_form_inv_double = t_form_3x3.inv(); 
    
    // Force the inverted matrix back to a continuous, 32-bit float layout
    cv::Mat t_form_inv;
    t_form_inv_double.convertTo(t_form_inv, CV_32F);
    t_form_inv = t_form_inv.clone();

    cv::Mat cropped_person;
    cv::warpAffine(original_frame, cropped_person, t_form, cv::Size(bp_config.input_w, bp_config.input_h));
    
    cv::Mat blob = cv::dnn::blobFromImage(cropped_person, 1.0/255.0, cv::Size(), cv::Scalar(0,0,0), true, false);

    //Fix k_inv layout continuity
    cv::Mat k_inv_float;
    geo.cameraMatrixInverse.convertTo(k_inv_float, CV_32F);
    k_inv_float = k_inv_float.clone(); // Guarantees a continuous 9-element float array

    cudaMemcpyAsync(bp_ctx.d_input0, blob.ptr<float>(), blob.total() * sizeof(float), cudaMemcpyHostToDevice, bp_ctx.stream);
    cudaMemcpyAsync(bp_ctx.d_k_inv, k_inv_float.ptr<float>(), 9 * sizeof(float), cudaMemcpyHostToDevice, bp_ctx.stream);
    cudaMemcpyAsync(bp_ctx.d_t_form_inv, t_form_inv.ptr<float>(), 9 * sizeof(float), cudaMemcpyHostToDevice, bp_ctx.stream);
    
    // Run Inference First
    bp_ctx.context->enqueueV3(bp_ctx.stream);

    // Copy Outputs after inference is queued
    cudaMemcpyAsync(bp_ctx.h_pose25d.data(), bp_ctx.d_pose25d, bp_ctx.h_pose25d.size() * sizeof(float), cudaMemcpyDeviceToHost, bp_ctx.stream);
    cudaMemcpyAsync(bp_ctx.h_pose3d.data(), bp_ctx.d_pose3d, bp_ctx.h_pose3d.size() * sizeof(float), cudaMemcpyDeviceToHost, bp_ctx.stream);
    cudaMemcpyAsync(bp_ctx.h_pose2d.data(), bp_ctx.d_pose2d, bp_ctx.h_pose2d.size() * sizeof(float), cudaMemcpyDeviceToHost, bp_ctx.stream);
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

int pose_estimation::setup(std::string engine_file, std::string onnx_file, cv::Size targetSize){
	//load calibration data needed
	//handle engine creation
	compileOnnxToEngine(onnx_file, engine_file, targetSize);
	initializeBodyPose3D(engine_file);
	std::cout << "body pose runner setup succesfull" << std::endl;
	return 1;
}

//add reference to output data 
int pose_estimation::run(cv::Mat& original_frame,cv::Rect& person_box){
	processAndRunBodyPose(original_frame, person_box);
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
