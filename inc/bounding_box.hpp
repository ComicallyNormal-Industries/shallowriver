#include <vector>
#include <string>
#include <memory>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <unistd.h>

struct ModelConfig {
    int grid_h = 34;
    int grid_w = 60;
    int num_classes = 3;
    float conf_threshold = 0.40f;
    float nms_threshold = 0.45f;
    float stride_x = 16.0f;
    float stride_y = 16.0f;
    float bbox_norm_x = 35.0f;
    float bbox_norm_y = 35.0f;
    std::vector<std::string> class_labels = {"Person", "Bag", "Face"};
    std::vector<cv::Scalar> class_colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};
};

struct TRTContext {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;

	// float* d_input = nullptr;
    // float* d_bbox = nullptr;
    // float* d_cov = nullptr;

    // size_t input_bytes;
};

class bounding_box {
	private:
		bool compileOnnxToEngine(const std::string& onnxPath, const std::string& enginePath, cv::Size targetSize);

		std::vector<char> loadEngineFile(const std::string& filename);

		bool initializeTRT(const std::string& engine_file, const cv::Size& resolution);

		void runInference(const cv::Mat& input_blob);
	
		void cleanupTRT();

	public:

		ModelConfig config;
		TRTContext trt_ctx;

		int setup(std::string engine_file, std::string onnx_file, cv::Size targetSize, bool rebuild);
		int run(cv::Mat& input_blob);

		TRTContext* getContextPtr();
		ModelConfig* getConfigPtr();

		void printTRTContext();

		bounding_box();
		
		~bounding_box();

};
