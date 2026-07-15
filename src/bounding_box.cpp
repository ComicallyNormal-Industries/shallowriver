// includes
#include "bounding_box.hpp"
#include "runner.hpp"
		
bool bounding_box::compileOnnxToEngine(const std::string& onnxPath, const std::string& enginePath, cv::Size targetSize) {
			
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


std::vector<char> bounding_box::loadEngineFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.good()) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

bool bounding_box::initializeTRT(const std::string& engine_file, const cv::Size& resolution, const ModelConfig& config, TRTContext& trt) {
	std::vector<char> engine_data = loadEngineFile(engine_file);
 	if (engine_data.empty()) return false;

	trt.runtime.reset(nvinfer1::createInferRuntime(gLogger));
   	trt.engine.reset(trt.runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    trt.context.reset(trt.engine->createExecutionContext());

    trt.h_bbox_output.resize(1 * (config.num_classes * 4) * config.grid_h * config.grid_w); 
    trt.h_cov_output.resize(1 * config.num_classes * config.grid_h * config.grid_w);
    trt.input_bytes = 1 * 3 * resolution.height * resolution.width * sizeof(float);
    
    cudaMalloc(&trt.d_input, trt.input_bytes);
    cudaMalloc(&trt.d_bbox, trt.h_bbox_output.size() * sizeof(float));
    cudaMalloc(&trt.d_cov, trt.h_cov_output.size() * sizeof(float));
    cudaStreamCreate(&trt.stream);

    trt.context->setTensorAddress("input_1:0", trt.d_input);
    trt.context->setTensorAddress("output_bbox/BiasAdd:0", trt.d_bbox);
    trt.context->setTensorAddress("output_cov/Sigmoid:0", trt.d_cov);

    return true;
}

void bounding_box::runInference(TRTContext& trt, const cv::Mat& input_blob) {
	cudaMemcpyAsync(trt.d_input, input_blob.ptr<float>(), trt.input_bytes, cudaMemcpyHostToDevice, trt.stream);
	trt.context->enqueueV3(trt.stream);
  	cudaMemcpyAsync(trt.h_bbox_output.data(), trt.d_bbox, trt.h_bbox_output.size() * sizeof(float), cudaMemcpyDeviceToHost, trt.stream);
    cudaMemcpyAsync(trt.h_cov_output.data(), trt.d_cov, trt.h_cov_output.size() * sizeof(float), cudaMemcpyDeviceToHost, trt.stream);
    cudaStreamSynchronize(trt.stream);
}

void bounding_box::cleanupTRT(TRTContext& trt) {
	if (trt.stream) cudaStreamDestroy(trt.stream);
	if (trt.d_input) cudaFree(trt.d_input);
	if (trt.d_bbox) cudaFree(trt.d_bbox);
    if (trt.d_cov) cudaFree(trt.d_cov);
}

bounding_box::bounding_box(){
	;
}

bounding_box::~bounding_box(){
	;
}



