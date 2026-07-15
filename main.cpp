#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <unistd.h>
#include <Eigen/Dense>
#include <cfloat>
#include <cmath>

// --- Globals & Structures ---






i



// --- Initialization Functions ---

void initPoseLogger(BodyPoseContext& ctx, const std::string& filename) {
    // std::ios::out | std::ios::trunc wipes the file if it exists, creates if it doesn't
    ctx.poseFile.open(filename, std::ios::out | std::ios::trunc);
    if (!ctx.poseFile.is_open()) {
        std::cerr << "Error: Could not open 3dpose.txt for writing!" << std::endl;
    }
}














// --- Execution & Processing Functions ---








// Fixed to return nms_indices

float median(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}




// --- Main Pipeline ---

int main() {
    std::string onnx_file = "resnet34_peoplenet.onnx";
    std::string engine_file = "peoplenet.engine";

    ModelConfig config;
    TRTContext trt_ctx;

    // Check & Compile Engine PeopleNet
    if (access(engine_file.c_str(), F_OK) == -1) {
        std::cout << "Notice: Compiled execution target file '" << engine_file << "' not found." << std::endl;
        if (!compileOnnxToEngine(onnx_file, engine_file, peoplenet_resolution)) {
            return -1;
        }
    }



    // Initialize TensorRT Runtime
    if (!initializeTRT(engine_file, peoplenet_resolution, config, trt_ctx)) {
        std::cerr << "Error: Failed to initialize TensorRT." << std::endl;
        return -1;
    }
    
    // Check & Compile Engine bodypose
    std::string bp_onnx_file = "bodypose3dnet_performance.onnx"; 
    std::string bp_engine_file = "bodypose3dnet_performance.engine"; 
    
    if (access(bp_engine_file.c_str(), F_OK) == -1) {
        std::cout << "Compiling BodyPose3D Engine..." << std::endl;
        if (!compileOnnxToEngine(bp_onnx_file, bp_engine_file, cv::Size(192, 256))) return -1;
    }

    BodyPoseConfig bp_config;
    BodyPoseContext bp_ctx;
    if (!initializeBodyPose3D(bp_engine_file, bp_config, bp_ctx)) {
        std::cerr << "Failed to initialize BodyPose3D engine." << std::endl;
        return -1;
    }

    initPoseLogger(bp_ctx, "3dpose.txt");

    
    


    cleanupTRT(trt_ctx);
    cleanupBodyPose3D(bp_ctx); // Corrected cleanup function for BodyPose

    if (bp_ctx.poseFile.is_open()) {
        bp_ctx.poseFile.close();
    }
    return 0;
}
