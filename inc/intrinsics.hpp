#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <iostream>
#include <vector>

// calculate camera intrensics to remove lens distortion for optimal 2d and 3d outputs
namespace intrinsics {

	int run_calibration(int camera_id);
	int save_intrinsics(int camera_id, cv::Mat cameraMatrix, cv::Mat distCoeffs);
	int generate_charuco();
}
