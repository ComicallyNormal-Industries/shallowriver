#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <unistd.h>

struct CameraGeometry {
    cv::Mat cameraMatrixOrig;
    cv::Mat cameraMatrixScaled;
    cv::Mat cameraMatrixInverse;
    cv::Mat distortionCoeffs;
};

struct BodyPoseConfig {
		
	int input_w = 192;
	int input_h = 256;
	int num_keypoints = 34;
    
	// NVIDIA standard limb proportions
	std::vector<float> scale_normalized_mean_limb_lengths = {
		0.5000, 0.5000, 1.0000, 0.8175, 0.9889, 0.2610, 0.7942, 0.5724, 0.5078,
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3433, 0.8171,
		0.9912, 0.2610, 0.8259, 0.5724, 0.5078, 0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.3422, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
	};

	std::vector<float> mean_limb_lengths = {
		246.3427f, 246.3427f, 492.6854f, 402.4380f, 487.0321f, 128.6856f, 391.6295f,
		281.9928f, 249.9478f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
		0.0000f,   0.0000f, 169.1832f, 402.2611f, 488.1824f, 128.6848f, 407.5836f,
		281.9897f, 249.9489f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
		0.0000f,   0.0000f, 168.6137f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
		0.0000f
	};
};
	
struct BodyPoseContext {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;

    // Inputs changed to float*
    float *d_input0 = nullptr, *d_k_inv = nullptr, *d_t_form_inv = nullptr;
    float *d_scale_norm_limb = nullptr, *d_mean_limb = nullptr;

    // Outputs changed to float*
    float *d_pose2d = nullptr, *d_pose2d_org = nullptr, *d_pose25d = nullptr, *d_pose3d = nullptr;

    // Host buffers for outputs completely removed!
};

class pose_estimation {
	private:

		bool compileOnnxToEngine(const std::string& onnxPath, const std::string& enginePath, cv::Size targetSize);

		bool initializeBodyPose3D(const std::string& engine_file);

		std::vector<char> loadEngineFile(const std::string& filename);
		
		void processAndRunBodyPose(const cv::Mat& blob, const cv::Mat& t_form_inv);

		void cleanupBodyPose3D(BodyPoseContext& trt);

	public:

		 CameraGeometry geo;

		BodyPoseConfig bp_config;
	    BodyPoseContext bp_ctx;

		int setup(std::string engine_file, std::string onnx_file, cv::Size targetSize, CameraGeometry& loaded_geo, bool rebuild);

		//add reference to output for return data
		int run(const cv::Mat& blob, const cv::Mat& t_form_inv);
	
		BodyPoseContext* getContextPtr();
		BodyPoseConfig* getConfigPtr();
		
		
		
		pose_estimation(CameraGeometry in_geo);

		~pose_estimation();
};


