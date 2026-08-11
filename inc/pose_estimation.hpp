#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <unistd.h>
#include <bounding_box.hpp>

#pragma once

// camera intrensics. set when camera intrensics are calibrated
struct CameraGeometry {
    cv::Mat cameraMatrixOrig;
    cv::Mat cameraMatrixScaled;
    cv::Mat cameraMatrixInverse;
    cv::Mat distortionCoeffs;
	cv::Mat cameraMatrixInverseFloat;
};

// body pose model context	
struct BodyPoseContext {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
};

// handles body pose model compilation and inference
struct pose_estimation {
	bool compileOnnxToEngine(const std::string& onnxPath, const std::string& enginePath, cv::Size targetSize);

	bool initializeBodyPose3D(const std::string& engine_file);

	std::vector<char> loadEngineFile(const std::string& filename);
	
	void processAndRunBodyPose(bb_context_packet& context_packet);

	void cleanupBodyPose3D(BodyPoseContext& trt);

	BodyPoseContext bp_ctx;

	int setup(std::string engine_file, std::string onnx_file, cv::Size targetSize, bool rebuild);

	BodyPoseContext* getContextPtr();

};


