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

constexpr size_t PEOPLENET_WIDTH = 960;  // 60 grid_w * 16 stride_x
constexpr size_t PEOPLENET_HEIGHT = 544; // 34 grid_h * 16 stride_y
constexpr size_t PEOPLENET_CHANNELS = 3;

constexpr size_t GRID_W = 60;
constexpr size_t GRID_H = 34;
constexpr size_t NUM_CLASSES = 3;

// 2. Element counts for the fixed-size arrays
constexpr size_t INPUT_ELEMENTS = PEOPLENET_CHANNELS * PEOPLENET_HEIGHT * PEOPLENET_WIDTH;
constexpr size_t BBOX_ELEMENTS = (NUM_CLASSES * 4) * GRID_H * GRID_W;
constexpr size_t COV_ELEMENTS = NUM_CLASSES * GRID_H * GRID_W;

constexpr size_t FRAME_X = 1920;
constexpr size_t FRAME_Y = 1080;

constexpr size_t TRT_PADDING = 1024;

// struct bb_context_packet
// {

//     // float d_input[INPUT_ELEMENTS + TRT_PADDING];
//     // float d_bbox[BBOX_ELEMENTS + TRT_PADDING];
//     // float d_cov[COV_ELEMENTS + TRT_PADDING];
//     alignas(256) float d_input[INPUT_ELEMENTS + TRT_PADDING];
//     alignas(256) float d_bbox[BBOX_ELEMENTS + TRT_PADDING];
//     alignas(256) float d_cov[COV_ELEMENTS + TRT_PADDING];

// 	uint64_t frame_id;
// 	std::vector<cv::Rect> bboxes;
//     std::vector<float> confidences;
//     std::vector<int> class_ids;
//     std::vector<int> nms_indices;

// 	cv::Mat raw_frame;
//     cv::Mat model_input;
// 	// cv::Mat input_blob;

// 	// size_t input_bytes = 1 * 3 * resolution.height * resolution.width * sizeof(float);

// 	bb_context_packet() {
//         // Pre-allocate resized BGR frame buffer (8-bit, 3 channels)
//         // model_input.create(PEOPLENET_HEIGHT, PEOPLENET_WIDTH, CV_8UC3);
// 		raw_frame.create(cv::Size(1920, 1080), CV_8UC3);

//         // Wrap d_input directly into NCHW format (1 x 3 x Height x Width)
//         int sizes[4] = {1, 3, PEOPLENET_HEIGHT, PEOPLENET_WIDTH};
//         model_input = cv::Mat(4, sizes, CV_32F, d_input);
//     }
	
// 	// Optional helper methods if you still need the byte counts elsewhere in your code
//     size_t get_input_bytes() const { return INPUT_ELEMENTS * sizeof(float); }
//     size_t get_bbox_bytes() const { return BBOX_ELEMENTS * sizeof(float); }
//     size_t get_cov_bytes() const { return COV_ELEMENTS * sizeof(float); }
// };

struct bb_context_packet
{
    uint64_t frame_id;
    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<int> nms_indices;

    cv::Mat raw_frame;
    cv::Mat model_input;

    // CUDA Pointers (No longer fixed-size arrays)
    float* d_input = nullptr;
    float* d_bbox = nullptr;
    float* d_cov = nullptr;

    bb_context_packet() {
        // 1. Allocate ONLY the float arrays in pinned hardware memory
        cudaHostAlloc((void**)&d_input, (INPUT_ELEMENTS + TRT_PADDING) * sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc((void**)&d_bbox, (BBOX_ELEMENTS + TRT_PADDING) * sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc((void**)&d_cov, (COV_ELEMENTS + TRT_PADDING) * sizeof(float), cudaHostAllocMapped);

        // 2. Clear out NaN garbage strictly for the GPU arrays
        std::memset(d_input, 0, (INPUT_ELEMENTS + TRT_PADDING) * sizeof(float));
        std::memset(d_bbox, 0, (BBOX_ELEMENTS + TRT_PADDING) * sizeof(float));
        std::memset(d_cov, 0, (COV_ELEMENTS + TRT_PADDING) * sizeof(float));

        // 3. Initialize OpenCV objects in standard CPU RAM
        raw_frame.create(cv::Size(1920, 1080), CV_8UC3);
        int sizes[4] = {1, 3, PEOPLENET_HEIGHT, PEOPLENET_WIDTH};
        model_input = cv::Mat(4, sizes, CV_32F, d_input);
    }
    
    // 4. Clean up CUDA memory safely when the packet is destroyed
    ~bb_context_packet() {
        if (d_input) cudaFreeHost(d_input);
        if (d_bbox) cudaFreeHost(d_bbox);
        if (d_cov) cudaFreeHost(d_cov);
    }

    // 5. Prevent accidental deep copies that would cause double-frees
    bb_context_packet(const bb_context_packet&) = delete;
    bb_context_packet& operator=(const bb_context_packet&) = delete;
};


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
	public:
		bool compileOnnxToEngine(const std::string& onnxPath, const std::string& enginePath, cv::Size targetSize);

		std::vector<char> loadEngineFile(const std::string& filename);

		bool initializeTRT(const std::string& engine_file, const cv::Size& resolution);

		bool runInference(bb_context_packet& bb_context);
	
		void cleanupTRT();


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
