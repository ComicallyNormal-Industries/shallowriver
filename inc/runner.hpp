#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "bounding_box.hpp"
#include "pose_estimation.hpp"

struct TRTContext;
struct ModelConfig;

struct NvAR_Point3f {
    float x, y, z;
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

		CameraGeometry geo;

		BodyPoseContext* bp_ctx_ptr = nullptr;
    	BodyPoseConfig* bp_cfg_ptr = nullptr;

		TRTContext* bb_ctx_ptr = nullptr;
		ModelConfig* bb_cfg_ptr = nullptr;

		bounding_box bbox_runner;
    	pose_estimation pose_runner;



		bool loadAndScaleIntrinsics(const std::string& filepath, cv::Size origSize, cv::Size targetSize, CameraGeometry& outGeo);

		cv::Mat preprocessFrame(const cv::Mat& frame, cv::Size target_resolution);

		void decodeDetections(const TRTContext& trt, const ModelConfig& cfg, std::vector<cv::Rect>& bboxes, std::vector<float>& confidences, std::vector<int>& class_ids);

		std::vector<int> applyNMSAndRender(cv::Mat& output_image, const ModelConfig& cfg, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences, const std::vector<int>& class_ids);

		std::vector<NvAR_Point3f> processBodyPoseOutput(const std::vector<float>& pose25d, const std::vector<float>& pose3d_raw, int numKeypoints, const cv::Rect& person_box, int crop_w, int crop_h, const cv::Mat& cameraMatrix);

		int setup();

	public:

		int run();
		runner();
};
