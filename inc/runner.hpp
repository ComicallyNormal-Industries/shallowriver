#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "bounding_box.hpp"
#include "pose_estimation.hpp"
#include "text_logger.hpp"
#include <chrono>
#include <thread>
#include <atomic>
#include "bounded_queue.hpp"

struct FramePacket {
    uint64_t frame_id;
    cv::Mat raw_frame;
};

struct BBoxPacket {
    uint64_t frame_id;
    cv::Mat raw_frame;
    cv::Mat model_input;
    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<int> nms_indices;
};

struct RenderPacket {
    uint64_t frame_id;
    cv::Mat final_frame;
};

struct bb_context_in
{
	float* d_input = nullptr;
	size_t input_bytes;
};

struct bb_context_out
{
	float* d_bbox = nullptr;
    float* d_cov = nullptr;
	size_t input_bytes;
};

struct bp_context_in
{
	// Inputs changed to float*
    float *d_input0 = nullptr, *d_k_inv = nullptr, *d_t_form_inv = nullptr;
    float *d_scale_norm_limb = nullptr, *d_mean_limb = nullptr;
};

struct bp_context_out
{
	// Outputs changed to float*
    float *d_pose2d = nullptr, *d_pose2d_org = nullptr, *d_pose25d = nullptr, *d_pose3d = nullptr;
};


class runner {

	private:

		cv::VideoCapture cap;
		cv::Size stream_resolution;
	    cv::Size peoplenet_resolution;
    	std::string bb_onnx_file;
    	std::string bb_engine_file;
	    std::string bp_onnx_file;
	    std::string bp_engine_file;

		std::string calib_file;
		std::string text_log_file;


		CameraGeometry geo;

		BodyPoseContext* bp_ctx_ptr = nullptr;
    	BodyPoseConfig* bp_cfg_ptr = nullptr;

		TRTContext* bb_ctx_ptr = nullptr;
		ModelConfig* bb_cfg_ptr = nullptr;

		bounding_box bbox_runner;
    	pose_estimation pose_runner;

		pose_logger p_logger;

		std::chrono::high_resolution_clock::time_point last_frame_time;
    	float current_fps = 0.0f;

		std::atomic<bool> running{true};

		bool loadAndScaleIntrinsics(const std::string& filepath, cv::Size origSize, cv::Size targetSize, CameraGeometry& outGeo);

		cv::Mat preprocessFrame(const cv::Mat& frame, cv::Mat& out_model_input, cv::Size target_resolution);

		void decodeDetections(const std::vector<float>& safe_cov, const std::vector<float>& safe_bbox, const ModelConfig& cfg, std::vector<cv::Rect>& bboxes, std::vector<float>& confidences, std::vector<int>& class_ids);
		
		std::vector<int> applyNMS( const ModelConfig& cfg, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences);

		void renderDetections(cv::Mat& output_image, const ModelConfig& cfg, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences, const std::vector<int>& class_ids, const std::vector<int>& nms_indices);

		void preprocessBodyPoseInput(const cv::Mat& original_frame, const cv::Rect& person_box, int input_w, int input_h, cv::Mat& out_blob, cv::Mat& out_t_form_inv);

		// Inside inc/runner.hpp
		std::vector<NvAR_Point3f> processBodyPoseOutput(const float* pose25d, const float* pose3d_raw, int numKeypoints, const cv::Rect& person_box, int crop_w, int crop_h, const cv::Mat& cameraMatrix);

		int setup(int mode);

		SPSCLatestValue<FramePacket>  q1_2{};
    	BoundedQueue<BBoxPacket>   q2_3{3};
    	BoundedQueue<RenderPacket> q3_4{3};

		void stage1_capture();
    	void stage2_bbox();
    	void stage3_pose();
    	void stage4_output();
	
		public:

		int run(int mode);
		runner();
};
