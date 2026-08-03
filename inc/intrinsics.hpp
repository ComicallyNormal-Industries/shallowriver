#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <iostream>
#include <vector>

namespace intrinsics {

	int run_calibration(int camera_id);
	int save_intrinsics(int camera_id, cv::Mat cameraMatrix, cv::Mat distCoeffs);
	int generate_charuco();
}
