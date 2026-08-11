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

// runner manages the four pipeline stages and applying calibration for the cameras

struct runner {

	// opencv capture streams
	cv::VideoCapture cap1;
	cv::VideoCapture cap2;

	// frame size for the camera stream
	cv::Size stream_resolution;
	//frame size for the peoplenet model
	cv::Size peoplenet_resolution;
	
	// paths to model files
	std::string bb_onnx_file;
	std::string bb_engine_file;
	std::string bp_onnx_file;
	std::string bp_engine_file;

	// calibration and loging paths
	std::string calib_file1;
	std::string calib_file2;
	std::string text_log_file_1;
	std::string text_log_file_2;
	std::string perf_log_file_1;
	std::string perf_log_file_2;

	bool enable_perf_logging = false;
	bool enable_print_stats = false;

	// camera calibration geometry intrensics
	CameraGeometry geo1;
	CameraGeometry geo2;

	// body pose stream context pointer
	BodyPoseContext* bp_ctx_ptr = nullptr;

	// bounding box context pointer 
	BoundingBoxContext* bb_ctx_ptr = nullptr;

	// model config pointer for bounding box
	ModelConfig* bb_cfg_ptr = nullptr;

	// bounding box class used to run the bounding box model
	bounding_box bbox_runner;
	// pose estimation class used to run the peoplenet model
	pose_estimation pose_runner;

	// logs
	pose_logger p_logger_1;
	pose_logger p_logger_2;

	perf_logger p_perf_1;
	perf_logger p_perf_2;

	// performance logging timer
	std::chrono::high_resolution_clock::time_point last_frame_time;
	float current_fps = 0.0f;

	// handles shutdown of stages
	std::atomic<bool> running{true};

	// loads and calculates camera intrensics
	bool loadAndScaleIntrinsics(const std::string& filepath, cv::Size origSize, cv::Size targetSize, CameraGeometry& outGeo);

	// process the frame so it is ready for the bounding box model
	void preprocessFrame(const cv::Mat& frame, cv::Size target_resolution, cv::Mat& input_model, bb_context_packet& bb_context);

	// turns bounding box wieghts into boxes
	void decodeDetections(const ModelConfig& cfg, bb_context_packet& bb_context);
	
	// merge boxes into one box per person
	std::vector<int> applyNMS( const ModelConfig& cfg, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences);

	// renders the bounding boxes onto the output image
	void renderDetections(cv::Mat& output_image, const ModelConfig& cfg, bb_context_packet& bb_context, const std::vector<int>& nms_indices);

	// processes frame to use for the body pose model
	void preprocessBodyPoseInput(const cv::Mat& original_frame, const cv::Rect& person_box, int input_w, int input_h, float* h_input0_ptr, cv::Mat& out_t_form_inv);

	// run body pose model on each bounding box
	std::vector<NvAR_Point3f> processBodyPoseOutput(const float* pose25d, const float* pose3d_raw, int numKeypoints, const cv::Rect& person_box, int crop_w, int crop_h, const cv::Mat& cameraMatrix);

	// initialization function
	int setup(int mode, int camera_mode, bool enable_logging = false);

	// multi producer single consumer queue. Merges input frames using a round robin buffer to prevent starvation
	SPSCLatestValueCudaMulti<PacketPtr>  q1_2{};
	//single producer single consumer queue. Handles bounding box algorithms and model
	SPSCLatestValueCuda<PacketPtr>   q2_3{};
	//single producer single consumer queue. Handles body pose algorithms and model. Has gpu priority to prevent starvation
	SPSCLatestValueCuda<PacketPtr> q3_4{};

	// four stages of the pipeline

	// captures frames for each camera
	void stage1_capture();
	void stage1_capture2();
	// bounding box
	void stage2_bbox();
	// body pose
	void stage3_pose();
	// output and rendering of frames to users
	void stage4_output();

	// start the pipeline, quits on ESC key
	int run(int mode, int camera_id, bool enable_logging = false, bool enable_print = false);

};
