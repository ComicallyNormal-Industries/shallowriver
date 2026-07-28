#include "runner.hpp"

bool runner::loadAndScaleIntrinsics(const std::string& filepath, cv::Size origSize, cv::Size targetSize, CameraGeometry& outGeo) {
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

cv::Mat runner::preprocessFrame(const cv::Mat& frame, cv::Mat& out_model_input, cv::Size target_resolution) {
    cv::resize(frame, out_model_input, target_resolution);
    return cv::dnn::blobFromImage(out_model_input, 1.0 / 255.0, target_resolution, cv::Scalar(0,0,0), true, false);
}


void runner::decodeDetections(const TRTContext& trt, const ModelConfig& cfg, std::vector<cv::Rect>& bboxes, std::vector<float>& confidences, std::vector<int>& class_ids) {

    int stride_spatial = cfg.grid_h * cfg.grid_w;

    for (int c = 0; c < cfg.num_classes; ++c) {
        for (int y = 0; y < cfg.grid_h; ++y) {
            for (int x = 0; x < cfg.grid_w; ++x) {

                int cov_offset = (c * stride_spatial) + (y * cfg.grid_w) + x;
                
                // FIXED: Reading confidence directly from the Unified Memory pointer
                float confidence = trt.d_cov[cov_offset];

                if (confidence >= cfg.conf_threshold) {
                    int base_bbox_class = (c * 4);
                    int idx_x1 = ((base_bbox_class + 0) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_y1 = ((base_bbox_class + 1) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_x2 = ((base_bbox_class + 2) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_y2 = ((base_bbox_class + 3) * stride_spatial) + (y * cfg.grid_w) + x;

                    // FIXED: Reading bounding box coordinates directly from Unified Memory
                    float dx1 = trt.d_bbox[idx_x1];
                    float dy1 = trt.d_bbox[idx_y1];
                    float dx2 = trt.d_bbox[idx_x2];
                    float dy2 = trt.d_bbox[idx_y2];

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

std::vector<int> runner::applyNMS(const ModelConfig& cfg, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences) {
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(bboxes, confidences, cfg.conf_threshold, cfg.nms_threshold, nms_indices);
    return nms_indices;
}

void runner::renderDetections(cv::Mat& output_image, const ModelConfig& cfg, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences, const std::vector<int>& class_ids, const std::vector<int>& nms_indices) {
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
}

void runner::preprocessBodyPoseInput(const cv::Mat& original_frame, const cv::Rect& person_box, int input_w, int input_h, cv::Mat& out_blob, cv::Mat& out_t_form_inv) {
   
	//Calculate the scale factor to preserve the aspect ratio
    float scale = std::min(static_cast<float>(input_w) / person_box.width, 
                           static_cast<float>(input_h) / person_box.height);

    //Calculate the centering offsets (this creates the black padding)
    float scaled_w = person_box.width * scale;
    float scaled_h = person_box.height * scale;
    float dx = (input_w - scaled_w) / 2.0f;
    float dy = (input_h - scaled_h) / 2.0f;

    //Map the pixels from the original frame into the padded 192x256 target
    cv::Mat t_form = (cv::Mat_<float>(2, 3) << 
        scale, 0.0f, -person_box.x * scale + dx,
        0.0f, scale, -person_box.y * scale + dy
    );

    //Create the 3x3 Inverse Transform Matrix for the GPU
    cv::Mat t_form_3x3 = cv::Mat::eye(3, 3, CV_32F);
    t_form.copyTo(t_form_3x3(cv::Rect(0, 0, 3, 2))); 

    cv::Mat t_form_inv_double = t_form_3x3.inv(); 
    t_form_inv_double.convertTo(out_t_form_inv, CV_32F);
    out_t_form_inv = out_t_form_inv.clone();

    //warp the frame (OpenCV automatically pads empty space with black)
    cv::Mat cropped_person;
    cv::warpAffine(original_frame, cropped_person, t_form, cv::Size(input_w, input_h));
    
    //Convert to tensor blob
    out_blob = cv::dnn::blobFromImage(cropped_person, 1.0/255.0, cv::Size(), cv::Scalar(0,0,0), true, false);

}

// CHANGED: The first two arguments are now const float* instead of const std::vector<float>&
std::vector<NvAR_Point3f> runner::processBodyPoseOutput(
        const float* pose25d,
        const float* pose3d_raw,
        int numKeypoints,
        const cv::Rect& person_box,
        int crop_w,
        int crop_h,
        const cv::Mat& cameraMatrix) {

    std::vector<NvAR_Point3f> final_3d(numKeypoints);

    // Extract Camera Intrinsics
    float fx = cameraMatrix.at<double>(0, 0);
    float fy = cameraMatrix.at<double>(1, 1);
    float cx = cameraMatrix.at<double>(0, 2);
    float cy = cameraMatrix.at<double>(1, 2);

    for (int k = 0; k < numKeypoints; ++k) {
        // The array access works exactly the same on the raw pointers!
        float crop_x = pose25d[k * 4 + 0];
        float crop_y = pose25d[k * 4 + 1];

        float scale_x = static_cast<float>(person_box.width) / crop_w;
        float scale_y = static_cast<float>(person_box.height) / crop_h;
        float full_x = person_box.x + (crop_x * scale_x);
        float full_y = person_box.y + (crop_y * scale_y);

        float z_abs = pose3d_raw[k * 3 + 2];

        final_3d[k].x = (full_x - cx) * z_abs / fx;
        final_3d[k].y = (full_y - cy) * z_abs / fy;
        final_3d[k].z = z_abs;
    }

    return final_3d;
}

int runner::setup(int mode) {
	bool rebuild = false;
	if (mode == 2){
		rebuild = true;
	}
    stream_resolution = cv::Size(1920, 1080);
    peoplenet_resolution = cv::Size(960, 544);
    bb_onnx_file = "res/resnet34_peoplenet.onnx";
    bb_engine_file = "res/peoplenet.engine";
    bp_onnx_file = "res/bodypose3dnet_performance.onnx";
    bp_engine_file = "res/bodypose3dnet_performance.engine";

	calib_file = "res/calibration.yaml";

	text_log_file = "res/3d_key_points.txt";

    if (!loadAndScaleIntrinsics(calib_file, stream_resolution, peoplenet_resolution, geo)) {
        std::cerr << "Warning: Could not load calibration data." << std::endl;
    }
		
	cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));	
    cap.set(cv::CAP_PROP_FRAME_WIDTH, stream_resolution.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, stream_resolution.height);
	cap.set(cv::CAP_PROP_FPS, 30);

	std::cout << "set up bounding box runner" << std::endl;
	if(!bbox_runner.setup(bb_engine_file,bb_onnx_file, peoplenet_resolution, rebuild)){
		std::cout << "set up bounding box runner failed" << std::endl;
		return -1;
	}

	std::cout << "set up pose estimation runner" << std::endl;
	if(!pose_runner.setup(bp_engine_file,bp_onnx_file, cv::Size(192, 256), geo, rebuild)){
		std::cout << "set up pose estimation runner failed" << std::endl;
		return -1;
	}	

	bp_ctx_ptr = pose_runner.getContextPtr();
    bp_cfg_ptr = pose_runner.getConfigPtr();
		
    bb_ctx_ptr = bbox_runner.getContextPtr();
    bb_cfg_ptr = bbox_runner.getConfigPtr();

	p_logger.initPoseLogger(text_log_file);
	
	return 1;	
}

#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>

int runner::run(int mode) {
    if(!setup(mode)){
        std::cout << "runner setup failed" << std::endl;
        return -1;
    }

    std::cout << "Starting real-time production TensorRT 10 execution loop..." << std::endl;
    
    // Frame & processing variables
    cv::Mat frame, model_input, input_blob;
    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<int> nms_indices;
    cv::Mat blob, t_form_inv;

    // --- Profiling Accumulators (in milliseconds) ---
    double acc_cap = 0.0, acc_prep_bb = 0.0, acc_infer_bb = 0.0, acc_post_bb = 0.0;
    double acc_prep_pose = 0.0, acc_infer_pose = 0.0, acc_post_pose = 0.0;
    double acc_render_show = 0.0;
    int profile_frame_count = 0;
    const int PROFILE_INTERVAL = 30; // Print statistics every 30 frames

    while (cv::waitKey(1) != 27) { // Press ESC to terminate cleanly

        // 1. Camera Frame Capture Time
        auto t0 = std::chrono::high_resolution_clock::now();
        cap >> frame;
        auto t1 = std::chrono::high_resolution_clock::now();
        if (frame.empty()) { 
            std::cerr << "frame capture failed ... " << std::endl;
            break;
        }

        // FPS Calculation
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = current_time - last_frame_time;
        last_frame_time = current_time;
        float raw_fps = (elapsed.count() > 0.0f) ? (1.0f / elapsed.count()) : 0.0f;
        current_fps = (current_fps * 0.9f) + (raw_fps * 0.1f);
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps));

        // 2. Bounding Box Preprocessing Time
        auto t2 = std::chrono::high_resolution_clock::now();
        input_blob = preprocessFrame(frame, model_input, peoplenet_resolution);
        auto t3 = std::chrono::high_resolution_clock::now();

        // 3. Bounding Box Inference Time
        if(!bbox_runner.run(input_blob)){
            std::cout << "running bounding box model failed" << std::endl;
            return -1;
        }
        auto t4 = std::chrono::high_resolution_clock::now();

        // 4. Bounding Box Decoding & NMS Time
        decodeDetections(*bb_ctx_ptr, *bb_cfg_ptr, bboxes, confidences, class_ids);        
        nms_indices = applyNMS(*bb_cfg_ptr, bboxes, confidences);
        renderDetections(model_input, *bb_cfg_ptr, bboxes, confidences, class_ids, nms_indices);
        auto t5 = std::chrono::high_resolution_clock::now();

        // 5. Pose Estimation Loop Time
        double frame_prep_pose = 0.0;
        double frame_infer_pose = 0.0;
        double frame_post_pose = 0.0;

        for (int idx : nms_indices) {
            if (class_ids[idx] == 0) { // Check if it's a person
                cv::Rect person_box = bboxes[idx];
                
                person_box.x = std::max(0, person_box.x - 10);
                person_box.y = std::max(0, person_box.y - 10);
                person_box.width = std::min(model_input.cols - person_box.x, person_box.width + 20);
                person_box.height = std::min(model_input.rows - person_box.y, person_box.height + 20);

                // Pose Preprocess
                auto tp0 = std::chrono::high_resolution_clock::now();
                preprocessBodyPoseInput(model_input, person_box, bp_cfg_ptr->input_w, bp_cfg_ptr->input_h, blob, t_form_inv);
                auto tp1 = std::chrono::high_resolution_clock::now();

                // Pose Inference
                if(!pose_runner.run(blob, t_form_inv)){
                    std::cout << "running inference on body pose failed" << std::endl;
                }        
                auto tp2 = std::chrono::high_resolution_clock::now();

                // Pose Postprocess & Logging
                std::vector<NvAR_Point3f> final_3d = processBodyPoseOutput(
                    bp_ctx_ptr->d_pose25d,
                    bp_ctx_ptr->d_pose3d,
                    pose_runner.bp_config.num_keypoints,
                    person_box,
                    pose_runner.bp_config.input_w,
                    pose_runner.bp_config.input_h,
                    pose_runner.geo.cameraMatrixOrig
                );

                p_logger.log_keypoints(final_3d);

                for (int k = 0; k < bp_cfg_ptr->num_keypoints; ++k) {
                    float kx_crop = bp_ctx_ptr->d_pose2d[k * 3 + 0];
                    float ky_crop = bp_ctx_ptr->d_pose2d[k * 3 + 1];
                    float conf    = bp_ctx_ptr->d_pose2d[k * 3 + 2];        
    
                    if (conf > 0.5f) {
                        int actual_x = static_cast<int>(t_form_inv.at<float>(0, 0) * kx_crop + 
                                                        t_form_inv.at<float>(0, 1) * ky_crop + 
                                                        t_form_inv.at<float>(0, 2));

                        int actual_y = static_cast<int>(t_form_inv.at<float>(1, 0) * kx_crop + 
                                                        t_form_inv.at<float>(1, 1) * ky_crop + 
                                                        t_form_inv.at<float>(1, 2));
                
                        cv::circle(model_input, cv::Point(actual_x, actual_y), 4, cv::Scalar(0, 255, 255), -1);
                    }
                }
                auto tp3 = std::chrono::high_resolution_clock::now();

                frame_prep_pose  += std::chrono::duration<double, std::milli>(tp1 - tp0).count();
                frame_infer_pose += std::chrono::duration<double, std::milli>(tp2 - tp1).count();
                frame_post_pose  += std::chrono::duration<double, std::milli>(tp3 - tp2).count();
            }   
        }
        auto t6 = std::chrono::high_resolution_clock::now();

        // 6. Text Rendering & GUI Display Time
        cv::putText(
            model_input,
            fps_text, 
            cv::Point(15, 40),           
            cv::FONT_HERSHEY_SIMPLEX,    
            1.0,                         
            cv::Scalar(0, 255, 0),       
            2                            
        );

        cv::imshow("Active TensorRT 10 Framework Output", model_input);
        auto t7 = std::chrono::high_resolution_clock::now();

        // --- Accumulate durations ---
        acc_cap         += std::chrono::duration<double, std::milli>(t1 - t0).count();
        acc_prep_bb     += std::chrono::duration<double, std::milli>(t3 - t2).count();
        acc_infer_bb    += std::chrono::duration<double, std::milli>(t4 - t3).count();
        acc_post_bb     += std::chrono::duration<double, std::milli>(t5 - t4).count();
        acc_prep_pose   += frame_prep_pose;
        acc_infer_pose  += frame_infer_pose;
        acc_post_pose   += frame_post_pose;
        acc_render_show += std::chrono::duration<double, std::milli>(t7 - t6).count();

        profile_frame_count++;

        // --- Print Breakdown Summary Every 30 Frames ---
        if (profile_frame_count >= PROFILE_INTERVAL) {
            double avg_cap         = acc_cap / PROFILE_INTERVAL;
            double avg_prep_bb     = acc_prep_bb / PROFILE_INTERVAL;
            double avg_infer_bb    = acc_infer_bb / PROFILE_INTERVAL;
            double avg_post_bb     = acc_post_bb / PROFILE_INTERVAL;
            double avg_prep_pose   = acc_prep_pose / PROFILE_INTERVAL;
            double avg_infer_pose  = acc_infer_pose / PROFILE_INTERVAL;
            double avg_post_pose   = acc_post_pose / PROFILE_INTERVAL;
            double avg_render_show = acc_render_show / PROFILE_INTERVAL;
            double total_frame_ms  = avg_cap + avg_prep_bb + avg_infer_bb + avg_post_bb + 
                                     avg_prep_pose + avg_infer_pose + avg_post_pose + avg_render_show;

            std::cout << "\n================ [ PIPELINE TIMING REPORT ] ================\n"
                      << std::fixed << std::setprecision(2)
                      << " 1. Camera Capture (cap >> frame):      " << std::setw(6) << avg_cap         << " ms\n"
                      << " 2. BBox Preprocessing (CPU):           " << std::setw(6) << avg_prep_bb     << " ms\n"
                      << " 3. BBox Inference (GPU):               " << std::setw(6) << avg_infer_bb    << " ms\n"
                      << " 4. BBox Postprocessing (Decode/NMS):   " << std::setw(6) << avg_post_bb     << " ms\n"
                      << " 5. Pose Preprocessing (Crop/Warp):     " << std::setw(6) << avg_prep_pose   << " ms\n"
                      << " 6. Pose Inference (GPU):               " << std::setw(6) << avg_infer_pose  << " ms\n"
                      << " 7. Pose Postprocess & Disk Logging:    " << std::setw(6) << avg_post_pose   << " ms\n"
                      << " 8. GUI Render & Display (imshow):      " << std::setw(6) << avg_render_show << " ms\n"
                      << "------------------------------------------------------------\n"
                      << " TOTAL AVG FRAME TIME:                  " << std::setw(6) << total_frame_ms  << " ms "
                      << "(" << (1000.0 / total_frame_ms) << " FPS)\n"
                      << "============================================================\n" << std::endl;

            // Reset accumulators
            acc_cap = acc_prep_bb = acc_infer_bb = acc_post_bb = 0.0;
            acc_prep_pose = acc_infer_pose = acc_post_pose = 0.0;
            acc_render_show = 0.0;
            profile_frame_count = 0;
        }

        // Clear vectors
        bboxes.clear();
        confidences.clear();
        class_ids.clear();
    }

    return 0;
}

runner::runner() : pose_runner(geo), cap(0, cv::CAP_V4L2) {
    // The body can stay completely empty.
}
