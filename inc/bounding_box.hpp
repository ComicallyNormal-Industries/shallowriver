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
#pragma once

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


constexpr int input_w = 192;
constexpr int input_h = 256;
constexpr int num_keypoints = 34;

// NVIDIA standard limb proportions
constexpr std::array<float, 36> scale_normalized_mean_limb_lengths = {
    0.5000f, 0.5000f, 1.0000f, 0.8175f, 0.9889f, 0.2610f, 0.7942f, 0.5724f, 0.5078f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3433f, 0.8171f,
    0.9912f, 0.2610f, 0.8259f, 0.5724f, 0.5078f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.3422f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f
};

constexpr std::array<float, 36> mean_limb_lengths = {
    246.3427f, 246.3427f, 492.6854f, 402.4380f, 487.0321f, 128.6856f, 391.6295f,
    281.9928f, 249.9478f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f,   0.0000f, 169.1832f, 402.2611f, 488.1824f, 128.6848f, 407.5836f,
    281.9897f, 249.9489f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f,   0.0000f, 168.6137f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f
};

struct bb_context_packet
{
    uint64_t frame_id;
    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<int> nms_indices;

    cv::Mat raw_frame;
    cv::Mat model_input;
    //bdn
    //cv::Mat blob;

    // CUDA Pointers (No longer fixed-size arrays)
    float* d_input = nullptr;
    float* d_bbox = nullptr;
    float* d_cov = nullptr;


    float *d_input0 = nullptr, *d_k_inv = nullptr, *d_t_form_inv = nullptr;
    float *d_scale_norm_limb = nullptr, *d_mean_limb = nullptr;

    // Outputs changed to float*
    float *d_pose2d = nullptr, *d_pose2d_org = nullptr, *d_pose25d = nullptr, *d_pose3d = nullptr;

    bb_context_packet() {
        
        cudaMallocManaged((void**)&d_input, (INPUT_ELEMENTS + TRT_PADDING) * sizeof(float));
        cudaMallocManaged((void**)&d_bbox, (BBOX_ELEMENTS + TRT_PADDING) * sizeof(float));
        cudaMallocManaged((void**)&d_cov, (COV_ELEMENTS + TRT_PADDING) * sizeof(float));

        cudaMallocManaged((void**)&d_input0, 1 * 3 * input_h * input_w * sizeof(float));
        cudaMallocManaged((void**)&d_k_inv, 1 * 3 * 3 * sizeof(float));
        cudaMallocManaged((void**)&d_t_form_inv, 1 * 3 * 3 * sizeof(float));
        cudaMallocManaged((void**)&d_scale_norm_limb, 1 * 36 * sizeof(float));
        cudaMallocManaged((void**)&d_mean_limb, 1 * 36 * sizeof(float));

        cudaMallocManaged((void**)&d_pose2d, 1 * num_keypoints * 3 * sizeof(float));
        cudaMallocManaged((void**)&d_pose2d_org, 1 * num_keypoints * 3 * sizeof(float));
        cudaMallocManaged((void**)&d_pose25d, 1 * num_keypoints * 4 * sizeof(float));
        cudaMallocManaged((void**)&d_pose3d, 1 * num_keypoints * 3 * sizeof(float));

        // 2. Clear out NaN garbage strictly for the GPU arrays
        std::memset(d_input, 0, (INPUT_ELEMENTS + TRT_PADDING) * sizeof(float));
        std::memset(d_bbox, 0, (BBOX_ELEMENTS + TRT_PADDING) * sizeof(float));
        std::memset(d_cov, 0, (COV_ELEMENTS + TRT_PADDING) * sizeof(float));

        std::memset(d_pose2d, 0, 1 * num_keypoints * 3 * sizeof(float));
        std::memset(d_pose2d_org, 0, 1 * num_keypoints * 3 * sizeof(float));
        std::memset(d_pose25d, 0, 1 * num_keypoints * 4 * sizeof(float));
        std::memset(d_pose3d, 0, 1 * num_keypoints * 3 * sizeof(float));
        // 3. Initialize OpenCV objects in standard CPU RAM
        model_input.create(cv::Size(960, 544), CV_8UC3);

        // blob.create(cv::Size(960, 544), CV_8UC3);

        int sizes[4] = {1, 3, PEOPLENET_HEIGHT, PEOPLENET_WIDTH};
        //blob = cv::Mat(4, sizes, CV_32F, d_input);
    }
    
    // 4. Clean up CUDA memory safely when the packet is destroyed
    ~bb_context_packet() {
        if (d_input) cudaFree(d_input);
        if (d_bbox) cudaFree(d_bbox);
        if (d_cov) cudaFree(d_cov);

        // Body Pose Pointers
        if (d_input0) cudaFree(d_input0);
        if (d_k_inv) cudaFree(d_k_inv);
        if (d_t_form_inv) cudaFree(d_t_form_inv);
        if (d_scale_norm_limb) cudaFree(d_scale_norm_limb);
        if (d_mean_limb) cudaFree(d_mean_limb);
        
        if (d_pose2d) cudaFree(d_pose2d);
        if (d_pose2d_org) cudaFree(d_pose2d_org);
        if (d_pose25d) cudaFree(d_pose25d);
        if (d_pose3d) cudaFree(d_pose3d);

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

		bounding_box();
		
		~bounding_box();

};
