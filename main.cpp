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

// --- Globals & Structures ---

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

struct CameraGeometry {
    cv::Mat cameraMatrixOrig;
    cv::Mat cameraMatrixScaled;
    cv::Mat cameraMatrixInverse;
    cv::Mat distortionCoeffs;
};

struct ModelConfig {
    int grid_h = 34;
    int grid_w = 60;
    int num_classes = 3;
    float conf_threshold = 0.40f;
    float nms_threshold = 0.45f;
    float stride_x = 16.0f;
    float stride_y = 16.0f;
    float bbox_norm_x = 35.0f;
    float bbox_norm_y = 35.0f;
    std::vector<std::string> class_labels = {"Person", "Bag", "Face"};
    std::vector<cv::Scalar> class_colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};
};

struct TRTContext {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    
    void* d_input = nullptr;
    void* d_bbox = nullptr;
    void* d_cov = nullptr;
    
    std::vector<float> h_bbox_output;
    std::vector<float> h_cov_output;
    size_t input_bytes;
};

struct BodyPoseConfig {
    int input_w = 192;
    int input_h = 256;
    int num_keypoints = 34;
    
    // Placeholder for standard skeleton lengths. 
    // In production, these 36-element arrays come from NVIDIA's standard skeleton definition.
    std::vector<float> scale_normalized_mean_limb_lengths = std::vector<float>(36, 1.0f);
    std::vector<float> mean_limb_lengths = std::vector<float>(36, 1.0f);
};

struct BodyPoseContext {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    
    // Inputs
    void *d_input0 = nullptr, *d_k_inv = nullptr, *d_t_form_inv = nullptr;
    void *d_scale_norm_limb = nullptr, *d_mean_limb = nullptr;
    
    // Outputs
    void *d_pose2d = nullptr, *d_pose2d_org = nullptr, *d_pose25d = nullptr, *d_pose3d = nullptr;
    
    // Host buffers for outputs (Batch size of 1 for simplicity in this loop)
    std::vector<float> h_pose3d; 
    std::vector<float> h_pose2d_org;
    std::vector<float> h_pose2d;
    std::vector<float> h_pose25d;

    std::ofstream poseFile;
};

struct NvAR_Point3f {
    float x, y, z;
};
// --- Initialization Functions ---

void initPoseLogger(BodyPoseContext& ctx, const std::string& filename) {
    // std::ios::out | std::ios::trunc wipes the file if it exists, creates if it doesn't
    ctx.poseFile.open(filename, std::ios::out | std::ios::trunc);
    if (!ctx.poseFile.is_open()) {
        std::cerr << "Error: Could not open 3dpose.txt for writing!" << std::endl;
    }
}

bool loadAndScaleIntrinsics(const std::string& filepath, cv::Size origSize, cv::Size targetSize, CameraGeometry& outGeo) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open calibration file: " << filepath << ". Using identity matrices as fallback." << std::endl;
        // Fallback to prevent crash if matrix inverse is needed later
        outGeo.cameraMatrixOrig = cv::Mat::eye(3, 3, CV_64F);
        outGeo.cameraMatrixScaled = cv::Mat::eye(3, 3, CV_64F);
        outGeo.cameraMatrixInverse = cv::Mat::eye(3, 3, CV_64F);
        outGeo.distortionCoeffs = cv::Mat::zeros(1, 5, CV_64F);
        return false;
    }
    fs["camera_matrix"] >> outGeo.cameraMatrixOrig;
    fs["distortion_coefficients"] >> outGeo.distortionCoeffs;
    fs.release();

    double scale_x = static_cast<double>(targetSize.width) / origSize.width;
    double scale_y = static_cast<double>(targetSize.height) / origSize.height;

    outGeo.cameraMatrixScaled = outGeo.cameraMatrixOrig.clone();
    outGeo.cameraMatrixScaled.at<double>(0, 0) *= scale_x; 
    outGeo.cameraMatrixScaled.at<double>(0, 2) *= scale_x; 
    outGeo.cameraMatrixScaled.at<double>(1, 1) *= scale_y; 
    outGeo.cameraMatrixScaled.at<double>(1, 2) *= scale_y; 

    outGeo.cameraMatrixInverse = outGeo.cameraMatrixScaled.inv();
    return true;
}

bool compileOnnxToEngine(const std::string& onnxPath, const std::string& enginePath, cv::Size targetSize) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "TensorRT 10 Engine Compiler Active Engine Optimization" << std::endl;
    std::cout << "Building from: " << onnxPath << std::endl;
    std::cout << "========================================================\n" << std::endl;

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    if (!builder) return false;

    // Use strongly typed network configurations (TensorRT 10 native pattern)
    uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flags);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Critical Error: Failed to parse structural ONNX configuration layers." << std::endl;
        delete parser; delete network; delete builder;
        return false;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    
    int numInputs = network->getNbInputs();
    bool hasDynamic = false;

    for (int i = 0; i < numInputs; ++i) {
        nvinfer1::ITensor* input = network->getInput(i);
        nvinfer1::Dims dims = input->getDimensions();
        const char* inputName = input->getName();
        
        bool isInputDynamic = false;
        for (int d = 0; d < dims.nbDims; ++d) {
            if (dims.d[d] == -1) {
                isInputDynamic = true;
                hasDynamic = true;
            }
        }

        if (isInputDynamic) {
            nvinfer1::Dims minDims = dims;
            nvinfer1::Dims optDims = dims;
            nvinfer1::Dims maxDims = dims;

            if (minDims.d[0] == -1) {
                minDims.d[0] = 1; optDims.d[0] = 1; maxDims.d[0] = 1;
            }

            if (std::string(inputName) == "input0" || std::string(inputName) == "input_1:0") {
                if (dims.nbDims == 4) {
                    minDims.d[2] = targetSize.height; minDims.d[3] = targetSize.width;
                    optDims.d[2] = targetSize.height; optDims.d[3] = targetSize.width;
                    maxDims.d[2] = targetSize.height; maxDims.d[3] = targetSize.width;
                }
            } else {
                for (int d = 1; d < dims.nbDims; ++d) {
                    if (dims.d[d] == -1) {
                        minDims.d[d] = 3; optDims.d[d] = 3; maxDims.d[d] = 3;
                    }
                }
            }

            profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims);
            profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims);
            profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);
            std::cout << "Added optimization profile mapping for dynamic input: " << inputName << std::endl;
        }
    }

    // Assign the profile if it was requested. If not needed, builder ownership handles cleanup.
    if (hasDynamic) {
        config->addOptimizationProfile(profile);
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // Let the builder auto-optimize target arrays internally via strongly typed layout constraints
    std::cout << "Hardware mapping validation initialized..." << std::endl;

    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (!serializedModel) {
        std::cerr << "Critical Error: Model optimization engine generation failed." << std::endl;
        delete config; delete parser; delete network; delete builder;
        return false;
    }

    std::ofstream engineFile(enginePath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    engineFile.close();

    std::cout << "\n>>> Production Engine Compiled and Saved to Disk: " << enginePath << " <<<\n" << std::endl;

    delete serializedModel; delete config; delete parser; delete network; delete builder;
    return true;
}
std::vector<char> loadEngineFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.good()) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

bool initializeTRT(const std::string& engine_file, const cv::Size& resolution, const ModelConfig& config, TRTContext& trt) {
    std::vector<char> engine_data = loadEngineFile(engine_file);
    if (engine_data.empty()) return false;

    trt.runtime.reset(nvinfer1::createInferRuntime(gLogger));
    trt.engine.reset(trt.runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    trt.context.reset(trt.engine->createExecutionContext());

    trt.h_bbox_output.resize(1 * (config.num_classes * 4) * config.grid_h * config.grid_w); 
    trt.h_cov_output.resize(1 * config.num_classes * config.grid_h * config.grid_w);
    trt.input_bytes = 1 * 3 * resolution.height * resolution.width * sizeof(float);
    
    cudaMalloc(&trt.d_input, trt.input_bytes);
    cudaMalloc(&trt.d_bbox, trt.h_bbox_output.size() * sizeof(float));
    cudaMalloc(&trt.d_cov, trt.h_cov_output.size() * sizeof(float));
    cudaStreamCreate(&trt.stream);

    trt.context->setTensorAddress("input_1:0", trt.d_input);
    trt.context->setTensorAddress("output_bbox/BiasAdd:0", trt.d_bbox);
    trt.context->setTensorAddress("output_cov/Sigmoid:0", trt.d_cov);

    return true;
}

bool initializeBodyPose3D(const std::string& engine_file, const BodyPoseConfig& config, BodyPoseContext& trt) {
    std::vector<char> engine_data = loadEngineFile(engine_file);
    if (engine_data.empty()) return false;

    trt.runtime.reset(nvinfer1::createInferRuntime(gLogger));
    trt.engine.reset(trt.runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    trt.context.reset(trt.engine->createExecutionContext());

    cudaStreamCreate(&trt.stream);

    cudaMalloc(&trt.d_input0, 1 * 3 * config.input_h * config.input_w * sizeof(float));
    cudaMalloc(&trt.d_k_inv, 1 * 3 * 3 * sizeof(float));
    cudaMalloc(&trt.d_t_form_inv, 1 * 3 * 3 * sizeof(float));
    cudaMalloc(&trt.d_scale_norm_limb, 1 * 36 * sizeof(float));
    cudaMalloc(&trt.d_mean_limb, 1 * 36 * sizeof(float));

    cudaMalloc(&trt.d_pose2d, 1 * config.num_keypoints * 3 * sizeof(float));
    cudaMalloc(&trt.d_pose2d_org, 1 * config.num_keypoints * 3 * sizeof(float));
    cudaMalloc(&trt.d_pose25d, 1 * config.num_keypoints * 4 * sizeof(float));
    cudaMalloc(&trt.d_pose3d, 1 * config.num_keypoints * 3 * sizeof(float));

    trt.h_pose3d.resize(config.num_keypoints * 3);
    trt.h_pose2d_org.resize(config.num_keypoints * 3);
    trt.h_pose2d.resize(config.num_keypoints * 3);
    trt.h_pose25d.resize(config.num_keypoints * 4);

    trt.context->setTensorAddress("input0", trt.d_input0);
    trt.context->setTensorAddress("k_inv", trt.d_k_inv);
    trt.context->setTensorAddress("t_form_inv", trt.d_t_form_inv);
    trt.context->setTensorAddress("scale_normalized_mean_limb_lengths", trt.d_scale_norm_limb);
    trt.context->setTensorAddress("mean_limb_lengths", trt.d_mean_limb);
    
    trt.context->setTensorAddress("pose2d", trt.d_pose2d);
    trt.context->setTensorAddress("pose2d_org_img", trt.d_pose2d_org);
    trt.context->setTensorAddress("pose25d", trt.d_pose25d);
    trt.context->setTensorAddress("pose3d", trt.d_pose3d);

    cudaMemcpyAsync(trt.d_scale_norm_limb, config.scale_normalized_mean_limb_lengths.data(), 36 * sizeof(float), cudaMemcpyHostToDevice, trt.stream);
    cudaMemcpyAsync(trt.d_mean_limb, config.mean_limb_lengths.data(), 36 * sizeof(float), cudaMemcpyHostToDevice, trt.stream);
    cudaStreamSynchronize(trt.stream);

    return true;
}

void cleanupBodyPose3D(BodyPoseContext& trt) {
    if (trt.stream) cudaStreamDestroy(trt.stream);
    if (trt.d_input0) cudaFree(trt.d_input0);
    if (trt.d_k_inv) cudaFree(trt.d_k_inv);
    if (trt.d_t_form_inv) cudaFree(trt.d_t_form_inv);
    if (trt.d_scale_norm_limb) cudaFree(trt.d_scale_norm_limb);
    if (trt.d_mean_limb) cudaFree(trt.d_mean_limb);
    if (trt.d_pose2d) cudaFree(trt.d_pose2d);
    if (trt.d_pose2d_org) cudaFree(trt.d_pose2d_org);
    if (trt.d_pose25d) cudaFree(trt.d_pose25d);
    if (trt.d_pose3d) cudaFree(trt.d_pose3d);
}

void cleanupTRT(TRTContext& trt) {
    if (trt.stream) cudaStreamDestroy(trt.stream);
    if (trt.d_input) cudaFree(trt.d_input);
    if (trt.d_bbox) cudaFree(trt.d_bbox);
    if (trt.d_cov) cudaFree(trt.d_cov);
}

// --- Execution & Processing Functions ---

cv::Mat preprocessFrame(const cv::Mat& frame, cv::Size target_resolution) {
    cv::Mat model_input;
    cv::resize(frame, model_input, target_resolution);
    return cv::dnn::blobFromImage(model_input, 1.0 / 255.0, target_resolution, cv::Scalar(0,0,0), true, false);
}

void runInference(TRTContext& trt, const cv::Mat& input_blob) {
    cudaMemcpyAsync(trt.d_input, input_blob.ptr<float>(), trt.input_bytes, cudaMemcpyHostToDevice, trt.stream);
    trt.context->enqueueV3(trt.stream);
    cudaMemcpyAsync(trt.h_bbox_output.data(), trt.d_bbox, trt.h_bbox_output.size() * sizeof(float), cudaMemcpyDeviceToHost, trt.stream);
    cudaMemcpyAsync(trt.h_cov_output.data(), trt.d_cov, trt.h_cov_output.size() * sizeof(float), cudaMemcpyDeviceToHost, trt.stream);
    cudaStreamSynchronize(trt.stream);
}

void decodeDetections(const TRTContext& trt, const ModelConfig& cfg, 
                      std::vector<cv::Rect>& bboxes, std::vector<float>& confidences, std::vector<int>& class_ids) {
    int stride_spatial = cfg.grid_h * cfg.grid_w;

    for (int c = 0; c < cfg.num_classes; ++c) {
        for (int y = 0; y < cfg.grid_h; ++y) {
            for (int x = 0; x < cfg.grid_w; ++x) {
                
                int cov_offset = (c * stride_spatial) + (y * cfg.grid_w) + x;
                float confidence = trt.h_cov_output[cov_offset];

                if (confidence >= cfg.conf_threshold) {
                    int base_bbox_class = (c * 4);
                    int idx_x1 = ((base_bbox_class + 0) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_y1 = ((base_bbox_class + 1) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_x2 = ((base_bbox_class + 2) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_y2 = ((base_bbox_class + 3) * stride_spatial) + (y * cfg.grid_w) + x;

                    float dx1 = trt.h_bbox_output[idx_x1];
                    float dy1 = trt.h_bbox_output[idx_y1];
                    float dx2 = trt.h_bbox_output[idx_x2];
                    float dy2 = trt.h_bbox_output[idx_y2];

                    float cell_center_x = static_cast<float>(x) * cfg.stride_x + 0.5f;
                    float cell_center_y = static_cast<float>(y) * cfg.stride_y + 0.5f;

                    float x1 = cell_center_x - (dx1 * cfg.bbox_norm_x);
                    float y1 = cell_center_y - (dy1 * cfg.bbox_norm_y);
                    float x2 = cell_center_x + (dx2 * cfg.bbox_norm_x);
                    float y2 = cell_center_y + (dy2 * cfg.bbox_norm_y);

                    x1 = std::max(0.0f, std::min(x1, 959.0f));
                    y1 = std::max(0.0f, std::min(y1, 543.0f));
                    x2 = std::max(0.0f, std::min(x2, 959.0f));
                    y2 = std::max(0.0f, std::min(y2, 543.0f));

                    float width  = x2 - x1;
                    float height = y2 - y1;

                    if (width > 4.0f && height > 4.0f) {
                        bboxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(width), static_cast<int>(height)));
                        confidences.push_back(confidence);
                        class_ids.push_back(c);
                    }
                }
            }
        }
    }
}

// Fixed to return nms_indices
std::vector<int> applyNMSAndRender(cv::Mat& output_image, const ModelConfig& cfg, 
                       const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences, const std::vector<int>& class_ids) {
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(bboxes, confidences, cfg.conf_threshold, cfg.nms_threshold, nms_indices);

    for (int idx : nms_indices) {
        cv::Rect box = bboxes[idx];
        int class_id = class_ids[idx];
        float score = confidences[idx];

        cv::rectangle(output_image, box, cfg.class_colors[class_id], 2);

        std::string label = cfg.class_labels[class_id] + ": " + cv::format("%.2f", score);
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int top = std::max(box.y, label_size.height);
        cv::rectangle(output_image, cv::Point(box.x, top - label_size.height), cv::Point(box.x + label_size.width, top + baseLine), cfg.class_colors[class_id], cv::FILLED);
        cv::putText(output_image, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    return nms_indices;
}

std::vector<float> calculateZRoots(const std::vector<float>& X0, const std::vector<float>& X1,
    const std::vector<float>& Y0, const std::vector<float>& Y1,
    const std::vector<float>& Zrel0,
    const std::vector<float>& Zrel1, const std::vector<float>& C) {
    std::vector<float> zRoots(X0.size());
    for (size_t i = 0; i < X0.size(); i++) {
        double x0 = (double)X0[i], x1 = (double)X1[i], y0 = (double)Y0[i], y1 = (double)Y1[i],
            z0 = (double)Zrel0[i], z1 = (double)Zrel1[i];
        double a = ((x1 - x0) * (x1 - x0)) + ((y1 - y0) * (y1 - y0));
        double b = 2 * (z1 * ((x1 * x1) + (y1 * y1) - x1 * x0 - y1 * y0) +
                    z0 * ((x0 * x0) + (y0 * y0) - x1 * x0 - y1 * y0));
        double c = ((x1 * z1 - x0 * z0) * (x1 * z1 - x0 * z0)) +
                   ((y1 * z1 - y0 * z0) * (y1 * z1 - y0 * z0)) +
                   ((z1 - z0) * (z1 - z0)) - (C[i] * C[i]);
        double d = (b * b) - (4 * a * c);

        a = fmax(DBL_EPSILON, a);
        d = fmax(DBL_EPSILON, d);
        zRoots[i] = (float) ((-b + sqrt(d)) / (2 * a + 1e-8));
    }
    return zRoots;
}

float median(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

std::vector<NvAR_Point3f> liftKeypoints25DTo3D(
    const std::vector<float>& pose25d, // Expected size: numKeypoints * 4 [x,y,zRel,conf]
    int numKeypoints,
    const cv::Mat& cameraMatrixInverse,
    const std::vector<float>& limbLengths) {

    const int ROOT = 0;
    std::vector<float> zRel(numKeypoints, 0.f);
    Eigen::MatrixXf XY1(numKeypoints, 3);
    std::vector<float> C;

    // Standard limb connection indices used by NVIDIA bodypose3dnet
    std::vector<int> idx0 = { 0, 3, 6, 8, 5, 2, 2, 21, 23, 21, 7, 4, 1, 1, 20, 22, 20 };
    std::vector<int> idx1 = { 3, 6, 0, 5, 2, 0, 21, 23, 25, 6, 4, 1, 0, 20, 22, 24, 6 };

    std::vector<float> X0(idx0.size(), 0.f), Y0(idx0.size(), 0.f), X1(idx0.size(), 0.f), Y1(idx0.size(), 0.f),
        zRel0(idx0.size(), 0.f), zRel1(idx0.size(), 0.f);

    // Convert OpenCV inverse matrix to Eigen
    Eigen::Matrix3f KInv;
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            KInv(i,j) = cameraMatrixInverse.at<double>(i,j);

    for (int i = 0; i < numKeypoints; i++) {
        zRel[i] = pose25d[i * 4 + 2]; // Extract Z-relative from 2.5D pose
        XY1.row(i) << pose25d[i * 4 + 0], pose25d[i * 4 + 1], 1.f; // X, Y, 1

        if (limbLengths[i] > 0.f) C.push_back(limbLengths[i]);
    }
    zRel[ROOT] = 0.f;

    // Apply Camera Inverse Transform
    XY1 = XY1 * KInv;

    for (size_t i = 0; i < idx0.size(); i++) {
        X0[i] = XY1(idx0[i], 0); Y0[i] = XY1(idx0[i], 1);
        X1[i] = XY1(idx1[i], 0); Y1[i] = XY1(idx1[i], 1);
        zRel0[i] = zRel[idx0[i]]; zRel1[i] = zRel[idx1[i]];
    }

    std::vector<float> zRoots = calculateZRoots(X0, X1, Y0, Y1, zRel0, zRel1, C);
    float zRootsMedian = median(zRoots);

    std::vector<NvAR_Point3f> p3d(numKeypoints, { 0.f, 0.f, 0.f });
    for (int i = 0; i < numKeypoints; i++) {
        p3d[i].x = XY1(i, 0) * (zRel[i] + zRootsMedian);
        p3d[i].y = XY1(i, 1) * (zRel[i] + zRootsMedian);
        p3d[i].z = XY1(i, 2) * (zRel[i] + zRootsMedian);
    }
    return p3d;
}

void processAndRunBodyPose(const cv::Mat& original_frame, const cv::Rect& person_box, const CameraGeometry& geo, 
                           BodyPoseContext& bp_ctx, const BodyPoseConfig& bp_cfg) {
    
    cv::Point2f src_pts[3], dst_pts[3];
    src_pts[0] = cv::Point2f(person_box.x, person_box.y);
    src_pts[1] = cv::Point2f(person_box.x + person_box.width, person_box.y);
    src_pts[2] = cv::Point2f(person_box.x, person_box.y + person_box.height);
    
    dst_pts[0] = cv::Point2f(0, 0);
    dst_pts[1] = cv::Point2f(bp_cfg.input_w, 0);
    dst_pts[2] = cv::Point2f(0, bp_cfg.input_h);

    // --- Fix t_form_inv allocation and type precision ---
    cv::Mat t_form = cv::getAffineTransform(src_pts, dst_pts);
    cv::Mat t_form_3x3 = cv::Mat::eye(3, 3, CV_32F);
    t_form.convertTo(t_form_3x3(cv::Rect(0, 0, 3, 2)), CV_32F);
    
    // Assign to a temporary cv::Mat first to resolve the expression
    cv::Mat t_form_inv_double = t_form_3x3.inv(); 
    
    // Force the inverted matrix back to a continuous, 32-bit float layout
    cv::Mat t_form_inv;
    t_form_inv_double.convertTo(t_form_inv, CV_32F);
    t_form_inv = t_form_inv.clone();

    cv::Mat cropped_person;
    cv::warpAffine(original_frame, cropped_person, t_form, cv::Size(bp_cfg.input_w, bp_cfg.input_h));
    
    cv::Mat blob = cv::dnn::blobFromImage(cropped_person, 1.0/255.0, cv::Size(), cv::Scalar(0,0,0), true, false);

    // --- Fix k_inv layout continuity ---
    cv::Mat k_inv_float;
    geo.cameraMatrixInverse.convertTo(k_inv_float, CV_32F);
    k_inv_float = k_inv_float.clone(); // Guarantees a continuous 9-element float array

    cudaMemcpyAsync(bp_ctx.d_input0, blob.ptr<float>(), blob.total() * sizeof(float), cudaMemcpyHostToDevice, bp_ctx.stream);
    cudaMemcpyAsync(bp_ctx.d_k_inv, k_inv_float.ptr<float>(), 9 * sizeof(float), cudaMemcpyHostToDevice, bp_ctx.stream);
    cudaMemcpyAsync(bp_ctx.d_t_form_inv, t_form_inv.ptr<float>(), 9 * sizeof(float), cudaMemcpyHostToDevice, bp_ctx.stream);
    cudaMemcpyAsync(bp_ctx.h_pose25d.data(), bp_ctx.d_pose25d, bp_ctx.h_pose25d.size() * sizeof(float), cudaMemcpyDeviceToHost, bp_ctx.stream);

    bp_ctx.context->enqueueV3(bp_ctx.stream);

    // Copy the raw cropped coordinates instead of the engine's estimated original coordinates
    cudaMemcpyAsync(bp_ctx.h_pose3d.data(), bp_ctx.d_pose3d, bp_ctx.h_pose3d.size() * sizeof(float), cudaMemcpyDeviceToHost, bp_ctx.stream);
    cudaMemcpyAsync(bp_ctx.h_pose2d.data(), bp_ctx.d_pose2d, bp_ctx.h_pose2d.size() * sizeof(float), cudaMemcpyDeviceToHost, bp_ctx.stream);
    cudaStreamSynchronize(bp_ctx.stream);
}

// --- Main Pipeline ---

int main() {
    cv::Size stream_resolution(1920, 1080);
    cv::Size peoplenet_resolution(960, 544);
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

    // Load Calibration
    CameraGeometry geo;
    if (!loadAndScaleIntrinsics("calibration.yaml", stream_resolution, peoplenet_resolution, geo)) {
        std::cerr << "Warning: Could not load calibration data." << std::endl;
    }

    // Initialize TensorRT Runtime
    if (!initializeTRT(engine_file, peoplenet_resolution, config, trt_ctx)) {
        std::cerr << "Error: Failed to initialize TensorRT." << std::endl;
        return -1;
    }
    
    // Check & Compile Engine bodypose
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

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, stream_resolution.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, stream_resolution.height);

    std::cout << "Starting real-time production TensorRT 10 execution loop..." << std::endl;
    cv::Mat frame, model_input;

    while (cv::waitKey(1) != 27) { // Press ESC to terminate cleanly
        cap >> frame;
        if (frame.empty()) break;

        cv::resize(frame, model_input, peoplenet_resolution);
        cv::Mat input_blob = preprocessFrame(frame, peoplenet_resolution);

        runInference(trt_ctx, input_blob);

        std::vector<cv::Rect> bboxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        decodeDetections(trt_ctx, config, bboxes, confidences, class_ids);
        
        // Render boxes first
        std::vector<int> nms_indices = applyNMSAndRender(model_input, config, bboxes, confidences, class_ids);
    
        // NOW iterate over indices to define person_box and run keypoints
        for (int idx : nms_indices) {
            // Check if it's a person
            if (class_ids[idx] == 0) {
                cv::Rect person_box = bboxes[idx];
                
                // Keep the box within frame boundaries
                person_box.x = std::max(0, person_box.x - 10);
                person_box.y = std::max(0, person_box.y - 10);
                person_box.width = std::min(model_input.cols - person_box.x, person_box.width + 20);
                person_box.height = std::min(model_input.rows - person_box.y, person_box.height + 20);

                // Run inference on the crop
                processAndRunBodyPose(model_input, person_box, geo, bp_ctx, bp_config);

		// Lift the raw 2.5D output to True 3D world space coordinates
                std::vector<NvAR_Point3f> lifted3D = liftKeypoints25DTo3D(
                    bp_ctx.h_pose25d, 
                    bp_config.num_keypoints, 
                    geo.cameraMatrixInverse, 
                    bp_config.mean_limb_lengths
                );

                // Log the true 3D pose data to the file
                if (bp_ctx.poseFile.is_open()) {
                    bp_ctx.poseFile << "--- Frame Start ---" << std::endl;
                    for (int k = 0; k < bp_config.num_keypoints; ++k) {
                        bp_ctx.poseFile << "Keypoint_" << k << ": " 
                                        << lifted3D[k].x << ", " 
                                        << lifted3D[k].y << ", " 
                                        << lifted3D[k].z << std::endl;
                    }
                }

                // Draw keypoints inside the person loop
                for (int k = 0; k < bp_config.num_keypoints; ++k) {
                    float kx_crop = bp_ctx.h_pose2d[k * 3 + 0];
                    float ky_crop = bp_ctx.h_pose2d[k * 3 + 1];
                    float conf    = bp_ctx.h_pose2d[k * 3 + 2];
                    
                    if (conf > 0.3f) {
                        int actual_x = person_box.x + static_cast<int>((kx_crop / bp_config.input_w) * person_box.width);
                        int actual_y = person_box.y + static_cast<int>((ky_crop / bp_config.input_h) * person_box.height);
                        
                        cv::circle(model_input, cv::Point(actual_x, actual_y), 4, cv::Scalar(0, 255, 255), -1);
                    }
                }
            }
        }

        cv::imshow("Active TensorRT 10 Framework Output", model_input);
    }

    cleanupTRT(trt_ctx);
    cleanupBodyPose3D(bp_ctx); // Corrected cleanup function for BodyPose

    if (bp_ctx.poseFile.is_open()) {
        bp_ctx.poseFile.close();
    }
    return 0;
}
