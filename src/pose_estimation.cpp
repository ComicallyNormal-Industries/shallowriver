#include "pose_estimation.hpp"
#include "glogger.hpp"

bool pose_estimation::compileOnnxToEngine(const std::string& onnxPath, const std::string& enginePath, cv::Size targetSize) {
	std::cout << "\n========================================================" << std::endl;
	std::cout << "TensorRT 10 Engine Compiler Active Engine Optimization" << std::endl;
	std::cout << "Building from: " << onnxPath << std::endl;
	std::cout << "========================================================\n" << std::endl;

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	if (!builder) return false;
	
	uint32_t flags = 0; 
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

	if (builder->platformHasFastFp16()) {
        //config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 Hardware detected. Enabling FP16 optimization." << std::endl;
    } else {
        std::cout << "Warning: FP16 not supported on this device. Using FP32." << std::endl;
    }

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

    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&bp_ctx.stream, cudaStreamNonBlocking, greatestPriority);

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

void pose_estimation::processAndRunBodyPose(bb_context_packet& context_packet) {

    cudaMemcpyAsync(context_packet.d_input0, context_packet.h_input0, 1 * 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, bp_ctx.stream);

    bp_ctx.context->setTensorAddress("input0", context_packet.d_input0);
    bp_ctx.context->setTensorAddress("k_inv", context_packet.d_k_inv);
    bp_ctx.context->setTensorAddress("t_form_inv", context_packet.d_t_form_inv);
    bp_ctx.context->setTensorAddress("scale_normalized_mean_limb_lengths", context_packet.d_scale_norm_limb);
    bp_ctx.context->setTensorAddress("mean_limb_lengths", context_packet.d_mean_limb);

    bp_ctx.context->setTensorAddress("pose2d", context_packet.d_pose2d);
    bp_ctx.context->setTensorAddress("pose2d_org_img", context_packet.d_pose2d_org);
    bp_ctx.context->setTensorAddress("pose25d", context_packet.d_pose25d);
    bp_ctx.context->setTensorAddress("pose3d", context_packet.d_pose3d);

    bp_ctx.context->setInputShape("input0", nvinfer1::Dims4{1, 3, input_h, input_w});
    bp_ctx.context->setInputShape("k_inv", nvinfer1::Dims3{1, 3, 3});
    bp_ctx.context->setInputShape("t_form_inv", nvinfer1::Dims3{1, 3, 3});
    bp_ctx.context->setInputShape("scale_normalized_mean_limb_lengths", nvinfer1::Dims2{1, 36});
    bp_ctx.context->setInputShape("mean_limb_lengths", nvinfer1::Dims2{1, 36});

    if (!bp_ctx.context->enqueueV3(bp_ctx.stream)) {
        std::cerr << "Body pose 3D inference enqueue failed." << std::endl;
        return;
    }

    cudaMemcpyAsync(context_packet.h_pose2d, context_packet.d_pose2d, num_keypoints * 3 * sizeof(float), cudaMemcpyDeviceToHost, bp_ctx.stream);
    cudaMemcpyAsync(context_packet.h_pose25d, context_packet.d_pose25d, num_keypoints * 4 * sizeof(float), cudaMemcpyDeviceToHost, bp_ctx.stream);
    cudaMemcpyAsync(context_packet.h_pose3d, context_packet.d_pose3d, num_keypoints * 3 * sizeof(float), cudaMemcpyDeviceToHost, bp_ctx.stream);

    cudaStreamSynchronize(bp_ctx.stream);
}

void pose_estimation::cleanupBodyPose3D(BodyPoseContext& trt) {
    if (trt.stream) cudaStreamDestroy(trt.stream);
}

int pose_estimation::setup(std::string engine_file, std::string onnx_file, cv::Size targetSize, bool rebuild){
		
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

BodyPoseContext* pose_estimation::getContextPtr() { 
        return &bp_ctx; 
}
