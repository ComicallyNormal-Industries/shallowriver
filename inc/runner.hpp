#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "bounding_box.hpp"
#include "pose_estimation.hpp"

struct TRTContext;
struct ModelConfig;

struct CameraGeometry {
    cv::Mat cameraMatrixOrig;
    cv::Mat cameraMatrixScaled;
    cv::Mat cameraMatrixInverse;
    cv::Mat distortionCoeffs;
};

struct NvAR_Point3f {
    float x, y, z;
};

class TRTLogger : public nvinfer1::ILogger{
	public:
		void log(Severity severity, const char* msg) noexcept override;
};

inline TRTLogger gLogger;

class runner {

	private:
		bool loadAndScaleIntrinsics(const std::string& filepath, cv::Size origSize, cv::Size targetSize, CameraGeometry& outGeo);

		cv::Mat preprocessFrame(const cv::Mat& frame, cv::Size target_resolution);

		void decodeDetections(const TRTContext& trt, const ModelConfig& cfg, std::vector<cv::Rect>& bboxes, std::vector<float>& confidences, std::vector<int>& class_ids);

		std::vector<int> applyNMSAndRender(cv::Mat& output_image, const ModelConfig& cfg, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences, const std::vector<int>& class_ids);

		std::vector<NvAR_Point3f> processBodyPoseOutput(const std::vector<float>& pose25d, const std::vector<float>& pose3d_raw, int numKeypoints, const cv::Rect& person_box, int crop_w, int crop_h, const cv::Mat& cameraMatrix);

		void setup();

	public:

		int run();

};
