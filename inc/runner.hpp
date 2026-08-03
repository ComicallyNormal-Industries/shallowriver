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
#include "camera_defs.hpp"
#pragma once

class runner {

	public:

		cv::VideoCapture cap1;
		cv::VideoCapture cap2;
		bool multicam;
		cv::Size stream_resolution;
	    cv::Size peoplenet_resolution;
    	std::string bb_onnx_file;
    	std::string bb_engine_file;
	    std::string bp_onnx_file;
	    std::string bp_engine_file;

		std::string calib_file1;
		std::string calib_file2;
		std::string text_log_file;

		//std::string gst_pipeline = "v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";

		CameraGeometry geo1;
		CameraGeometry geo2;


		BodyPoseContext* bp_ctx_ptr = nullptr;

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

		void preprocessBodyPoseInput(const cv::Mat& original_frame, const cv::Rect& person_box, int input_w, int input_h, float* d_input0_ptr, cv::Mat& out_t_form_inv);

		std::vector<NvAR_Point3f> processBodyPoseOutput(const float* pose25d, const float* pose3d_raw, int numKeypoints, const cv::Rect& person_box, int crop_w, int crop_h, const cv::Mat& cameraMatrix);

		int setup(int mode, int camera_mode);

		SPSCLatestValueCuda<PacketPtr>  q1_2{};
    	SPSCLatestValueCuda<PacketPtr>   q2_3{};
    	SPSCLatestValueCuda<PacketPtr> q3_4{};

		void stage1_capture();
    	void stage2_bbox();
    	void stage3_pose();
    	void stage4_output();

		int run(int mode, int camera_id);

};
