// includes
#include "bounding_box.hpp"
#include "glogger.hpp"
		
bool bounding_box::compileOnnxToEngine(const std::string& onnxPath, const std::string& enginePath, cv::Size targetSize) {
			
	std::cout << "\n========================================================" << std::endl;
	std::cout << "TensorRT 10 Engine Compiler Active Engine Optimization" << std::endl;
	std::cout << "Building from: " << onnxPath << std::endl;
    std::cout << "This will take about 10-15 minutes" << std::endl;
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

std::vector<char> bounding_box::loadEngineFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.good()) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

bool bounding_box::initializeTRT(const std::string& engine_file, const cv::Size& resolution) {
    std::vector<char> engine_data = loadEngineFile(engine_file);
    if (engine_data.empty()) return false;

    bb_ctx.runtime.reset(nvinfer1::createInferRuntime(gLogger));
    bb_ctx.engine.reset(bb_ctx.runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    bb_ctx.context.reset(bb_ctx.engine->createExecutionContext());
    

     int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&bb_ctx.stream, cudaStreamNonBlocking, leastPriority);

    return true;
}

bool bounding_box::runInference(bb_context_packet& bb_context) {
    // dynamically set addresses to the current frame in the pipeline
    bb_ctx.context->setTensorAddress("input_1:0", bb_context.d_input);
    bb_ctx.context->setTensorAddress("output_bbox/BiasAdd:0", bb_context.d_bbox);
    bb_ctx.context->setTensorAddress("output_cov/Sigmoid:0", bb_context.d_cov);

    static bool shapes_set = false;
    if (!shapes_set) {
        bb_ctx.context->setInputShape("input_1:0", nvinfer1::Dims4{1, 3, PEOPLENET_HEIGHT, PEOPLENET_WIDTH});
        shapes_set = true;
    }
    // run the model on gpu
    bb_ctx.context->enqueueV3(bb_ctx.stream);
    // wait until the gpu finishes to continue
    cudaError_t sync_status = cudaStreamSynchronize(bb_ctx.stream);

    if (sync_status != cudaSuccess) {
        std::cerr << "GPU Inference failed: " << cudaGetErrorString(sync_status) << std::endl; 
        return false;
    }
    
    return true;
}

void bounding_box::cleanupTRT() {
	if (bb_ctx.stream) cudaStreamDestroy(bb_ctx.stream);
}

int bounding_box::setup(std::string engine_file, std::string onnx_file, cv::Size targetSize, bool rebuild){
    // Check & Compile Engine PeopleNet
    if (access(engine_file.c_str(), F_OK) == -1 || rebuild) {
        std::cout << "Notice: Compiled execution target file '" << engine_file << "' not found." << std::endl;
        if (!compileOnnxToEngine(onnx_file, engine_file, targetSize)) {
            std::cout << "compileing bounding box engine file failed" << std::endl;
			return -1;
        }
    }

    // Initialize TensorRT Runtime
    if (!initializeTRT(engine_file, targetSize)) {
        std::cerr << "Error: Failed to initialize TensorRT." << std::endl;
        return -1;
    }
	return 1;

}

BoundingBoxContext* bounding_box::getContextPtr() { 
    return &bb_ctx; 
}

ModelConfig* bounding_box::getConfigPtr(){ 
	return &config; 
}




