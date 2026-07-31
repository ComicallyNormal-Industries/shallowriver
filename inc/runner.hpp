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
#pragma once

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

struct bp_context_packet
{
	// Inputs changed to float*
    float *d_input0 = nullptr, *d_k_inv = nullptr, *d_t_form_inv = nullptr;
    float *d_scale_norm_limb = nullptr, *d_mean_limb = nullptr;
	float *d_pose2d = nullptr, *d_pose2d_org = nullptr, *d_pose25d = nullptr, *d_pose3d = nullptr;
};



class runner {

	public:

		cv::VideoCapture cap;
		cv::Size stream_resolution;
	    cv::Size peoplenet_resolution;
    	std::string bb_onnx_file;
    	std::string bb_engine_file;
	    std::string bp_onnx_file;
	    std::string bp_engine_file;

		std::string calib_file;
		std::string text_log_file;

		std::string gst_pipeline = "v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";
		// std::string gst_pipeline = 
        // "v4l2src device=/dev/video0 ! "
        // "image/jpeg, width=1920, height=1080, framerate=30/1 ! "
        // "nvv4l2decoder mjpeg=1 ! "           // 1. Hardware decode the MJPEG
        // "nvvidconv ! "                       // 2. Hardware-accelerated memory copy out of NVMM
        // "video/x-raw, format=BGRx ! "        // 3. Output as 4-channel BGR (nvvidconv requirement)
        // "videoconvert ! "                    // 4. Standard CPU conversion
        // "video/x-raw, format=BGR ! "         // 5. Drop the alpha channel for standard OpenCV 3-channel
        // "appsink drop=true sync=false";

		    // gst_pipeline = 
        // "v4l2src device=/dev/video0 ! "
        // "image/jpeg, width=1920, height=1080, framerate=30/1 ! "
        // "nvjpegdec ! "
        // "video/x-raw ! "
        // "videoconvert ! "
        // "video/x-raw, format=BGR ! "
        // "appsink drop=true sync=false";

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

		void preprocessFrame(const cv::Mat& frame, cv::Size target_resolution, cv::Mat& input_model, bb_context_packet& bb_context);

		void decodeDetections(const ModelConfig& cfg, bb_context_packet& bb_context);
		
		std::vector<int> applyNMS( const ModelConfig& cfg, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences);

		void renderDetections(cv::Mat& output_image, const ModelConfig& cfg, bb_context_packet& bb_context, const std::vector<int>& nms_indices);

		//void preprocessBodyPoseInput(const cv::Mat& original_frame, const cv::Rect& person_box, int input_w, int input_h, cv::Mat& out_blob, cv::Mat& out_t_form_inv);
		void preprocessBodyPoseInput(const cv::Mat& original_frame, const cv::Rect& person_box, int input_w, int input_h, float* d_input0_ptr, cv::Mat& out_t_form_inv);

		// Inside inc/runner.hpp
		std::vector<NvAR_Point3f> processBodyPoseOutput(const float* pose25d, const float* pose3d_raw, int numKeypoints, const cv::Rect& person_box, int crop_w, int crop_h, const cv::Mat& cameraMatrix);

		int setup(int mode);

		SPSCLatestValueCuda<PacketPtr>  q1_2{};
    	SPSCLatestValueCuda<PacketPtr>   q2_3{};
    	SPSCLatestValueCuda<PacketPtr> q3_4{};

		void stage1_capture();
    	void stage2_bbox();
    	void stage3_pose();
    	void stage4_output();
	

		// x6<6> bb_context;
		SPSCLatestValueCuda<bb_context_packet> bp_context;

		int run(int mode);
		runner();
};
