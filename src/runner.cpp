#include "runner.hpp"

PacketPool global_pool;

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

void runner::preprocessFrame(const cv::Mat& frame, cv::Size target_resolution, cv::Mat& model_input, bb_context_packet& bb_context) {
    
    // 1. Resize the 2D image
    cv::resize(frame, model_input, target_resolution);
    
    // 2. Let OpenCV generate the 4D tensor in its own temporary CPU memory
    cv::Mat temp_blob;
    cv::dnn::blobFromImage(
        model_input,        // Input image
        temp_blob,          // Output array (OpenCV owns this temporary memory)
        1.0 / 255.0,        // Scale factor
        cv::Size(),         // Target size (already resized)
        cv::Scalar(0,0,0),  // Mean subtraction
        true,               // SwapRB (BGR to RGB)
        false,              // Crop
        CV_32F              // Depth
    );
    cudaMemcpy(bb_context.d_input, temp_blob.ptr<float>(), temp_blob.total() * sizeof(float), cudaMemcpyDefault);
}

void runner::decodeDetections(const ModelConfig& cfg, bb_context_packet& bb_context) {

    int stride_spatial = cfg.grid_h * cfg.grid_w;

    for (int c = 0; c < cfg.num_classes; ++c) {
        for (int y = 0; y < cfg.grid_h; ++y) {
            for (int x = 0; x < cfg.grid_w; ++x) {

                int cov_offset = (c * stride_spatial) + (y * cfg.grid_w) + x;
                
                float confidence = bb_context.d_cov[cov_offset];

                if (confidence >= cfg.conf_threshold) {
                    int base_bbox_class = (c * 4);
                    int idx_x1 = ((base_bbox_class + 0) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_y1 = ((base_bbox_class + 1) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_x2 = ((base_bbox_class + 2) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_y2 = ((base_bbox_class + 3) * stride_spatial) + (y * cfg.grid_w) + x;

                    float dx1 = bb_context.d_bbox[idx_x1];
                    float dy1 = bb_context.d_bbox[idx_y1];
                    float dx2 = bb_context.d_bbox[idx_x2];
                    float dy2 = bb_context.d_bbox[idx_y2];

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
                        bb_context.bboxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(width), static_cast<int>(height)));
                        bb_context.confidences.push_back(confidence);
                        bb_context.class_ids.push_back(c);
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

void runner::renderDetections(cv::Mat& output_image, const ModelConfig& cfg, bb_context_packet& bb_context, const std::vector<int>& nms_indices) {
    for (int idx : nms_indices) {
        cv::Rect box = bb_context.bboxes[idx];
        int class_id = bb_context.class_ids[idx];
        float score = bb_context.confidences[idx];

        cv::rectangle(output_image, box, cfg.class_colors[class_id], 2);

        std::string label = cfg.class_labels[class_id] + ": " + cv::format("%.2f", score);
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int top = std::max(box.y, label_size.height);
        cv::rectangle(output_image, cv::Point(box.x, top - label_size.height), cv::Point(box.x + label_size.width, top + baseLine), cfg.class_colors[class_id], cv::FILLED);
        cv::putText(output_image, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

void runner::preprocessBodyPoseInput(const cv::Mat& original_frame, const cv::Rect& person_box, int input_w, int input_h, float* d_input0_ptr, cv::Mat& out_t_form_inv) {
   
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

    //warp the frame (OpenCV automatically pads empty space with black)
    cv::Mat cropped_person;
    cv::warpAffine(original_frame, cropped_person, t_form, cv::Size(input_w, input_h));
    
    //Convert to tensor blob
    cv::Mat temp_blob;
    cv::dnn::blobFromImage(cropped_person, temp_blob, 1.0/255.0, cv::Size(), cv::Scalar(0,0,0), true, false, CV_32F);

    cudaMemcpy(d_input0_ptr, temp_blob.ptr<float>(), temp_blob.total() * sizeof(float), cudaMemcpyDefault);
}

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

// --- STAGE 1: Frame Capture ---
void runner::stage1_capture() {
    uint64_t frame_counter = 0;
    while (running) {
        auto t_start = std::chrono::steady_clock::now();
        
        cv::Mat frame;
        cap1 >> frame; 
        if (frame.empty()) break;

        auto t_end = std::chrono::steady_clock::now();
        double cap_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        PacketPtr p = get_pooled_packet();
        if (!p) continue; 

        p->t_cap = cap_time;
        p->frame_id = frame_counter++;
        frame.copyTo(p->raw_frame);
        p->bboxes.clear();
        p->confidences.clear();
        p->class_ids.clear();
        p->nms_indices.clear();

        q1_2.produce_update([&](PacketPtr& queue_slot) {
            queue_slot = std::move(p);
        });
    }
}

// --- STAGE 2: Model 1 (PeopleNet Bounding Box) ---
void runner::stage2_bbox() {
    cudaSetDevice(0); 
    while (true) {
        PacketPtr* in_slot = q1_2.wait_and_consume();
        if (in_slot == nullptr) continue;
        PacketPtr p = std::move(*in_slot);
        if (!p) continue; 

        auto t_stage_start = std::chrono::steady_clock::now();

        // Preprocess
        auto t0 = std::chrono::steady_clock::now();
        preprocessFrame(p->raw_frame, peoplenet_resolution, p->model_input, *p);
        p->t_s2_pre = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        // Inference
        t0 = std::chrono::steady_clock::now();
        bbox_runner.runInference(*p);
        p->t_s2_inf = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        // Decode & NMS
        t0 = std::chrono::steady_clock::now();
        decodeDetections(*bb_cfg_ptr, *p);
        p->nms_indices = applyNMS(*bb_cfg_ptr, p->bboxes, p->confidences);
        renderDetections(p->model_input, *bb_cfg_ptr, *p, p->nms_indices);
        p->t_s2_post = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        p->t_s2_total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_stage_start).count();

        q2_3.produce_update([&](PacketPtr& queue_slot) {
            queue_slot = std::move(p);
        });
    }
}

// --- STAGE 3: Model 2 (Body Pose 3D) ---
void runner::stage3_pose() {
    cudaSetDevice(0); 
    while (true) {
        PacketPtr* in_slot = q2_3.wait_and_consume();
        if (in_slot == nullptr) break; 
        PacketPtr p = std::move(*in_slot);
        if (!p) continue; 

        auto t_stage_start = std::chrono::steady_clock::now();
        p->t_s3_pre = 0.0;
        p->t_s3_inf = 0.0;
        p->t_s3_post = 0.0;

        for (int idx : p->nms_indices) {
            if (p->class_ids[idx] == 0) { 
                cv::Rect box = p->bboxes[idx];
                box.x = std::max(0, box.x - 10);
                box.y = std::max(0, box.y - 10);
                box.width = std::min(p->model_input.cols - box.x, box.width + 20);
                box.height = std::min(p->model_input.rows - box.y, box.height + 20);

                cv::Mat t_form_inv;
                
                // Preprocess Crop
                auto t0 = std::chrono::steady_clock::now();
                preprocessBodyPoseInput(p->model_input, box, input_w, input_h, p->d_input0, t_form_inv);
                p->t_s3_pre += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

                // Inference
                t0 = std::chrono::steady_clock::now();
                pose_runner.processAndRunBodyPose(*p);
                cudaDeviceSynchronize(); // MUST WAIT for accurate timing
                p->t_s3_inf += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

                // Decode & Draw
                t0 = std::chrono::steady_clock::now();
                std::vector<NvAR_Point3f> final_3d = processBodyPoseOutput(
                    p->d_pose25d, p->d_pose3d, num_keypoints, box, input_w, input_h, pose_runner.geo.cameraMatrixOrig
                );
                p_logger.log_keypoints(final_3d);

                for (int k = 0; k < num_keypoints; ++k) {
                    float kx = p->d_pose2d[k * 3 + 0];
                    float ky = p->d_pose2d[k * 3 + 1];
                    float conf = p->d_pose2d[k * 3 + 2];
                    if (conf > 0.5f) {
                        int ax = static_cast<int>(t_form_inv.at<float>(0, 0) * kx + t_form_inv.at<float>(0, 1) * ky + t_form_inv.at<float>(0, 2));
                        int ay = static_cast<int>(t_form_inv.at<float>(1, 0) * kx + t_form_inv.at<float>(1, 1) * ky + t_form_inv.at<float>(1, 2));
                        cv::circle(p->model_input, cv::Point(ax, ay), 4, cv::Scalar(0, 255, 255), -1);
                    }
                }
                p->t_s3_post += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
        }
        p->t_s3_total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_stage_start).count();

        q3_4.produce_update([&](PacketPtr& queue_slot) {
            queue_slot = std::move(p);
        });
    }
}

// --- STAGE 4: Output Display ---
void runner::stage4_output() {
    auto fps_start_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    double current_fps = 0.0;

    // Running sums for the 2-second benchmark report
    double b_cap = 0;
    double b_s2_tot = 0, b_s2_pre = 0, b_s2_inf = 0, b_s2_post = 0;
    double b_s3_tot = 0, b_s3_pre = 0, b_s3_inf = 0, b_s3_post = 0;

    while (true) {
        PacketPtr* in_slot = q3_4.wait_and_consume();
        if (in_slot == nullptr) break; 
        PacketPtr p = std::move(*in_slot);
        if (!p) continue; 

        frame_count++;
        
        // Accumulate times for this frame
        b_cap += p->t_cap;
        b_s2_tot += p->t_s2_total; b_s2_pre += p->t_s2_pre; b_s2_inf += p->t_s2_inf; b_s2_post += p->t_s2_post;
        b_s3_tot += p->t_s3_total; b_s3_pre += p->t_s3_pre; b_s3_inf += p->t_s3_inf; b_s3_post += p->t_s3_post;

        auto current_time = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(current_time - fps_start_time).count();

        // Print benchmark report every 2.0 seconds
        if (elapsed_seconds >= 2.0) {
            current_fps = frame_count / elapsed_seconds;
            
            std::cout << "\n=======================================================\n";
            std::cout << " Pipeline Benchmark Averages (" << current_fps << " FPS)\n";
            std::cout << "=======================================================\n";
            std::cout << "Stage 1 (V4L2 Capture):       " << (b_cap / frame_count) << " ms\n";
            std::cout << "-------------------------------------------------------\n";
            std::cout << "Stage 2 (BBox Net Total):     " << (b_s2_tot / frame_count) << " ms\n";
            std::cout << "  - Frame Preprocessing:      " << (b_s2_pre / frame_count) << " ms\n";
            std::cout << "  - TensorRT Inference:       " << (b_s2_inf / frame_count) << " ms\n";
            std::cout << "  - Decoding & NMS Merging:   " << (b_s2_post / frame_count) << " ms\n";
            std::cout << "-------------------------------------------------------\n";
            std::cout << "Stage 3 (Body Pose Total):    " << (b_s3_tot / frame_count) << " ms\n";
            std::cout << "  - ROI Preprocessing:        " << (b_s3_pre / frame_count) << " ms\n";
            std::cout << "  - TensorRT Inference:       " << (b_s3_inf / frame_count) << " ms\n";
            std::cout << "  - Decode Math & Draw:       " << (b_s3_post / frame_count) << " ms\n";
            std::cout << "=======================================================\n";
            
            // Reset aggregators for the next 2-second window
            fps_start_time = current_time;
            frame_count = 0;
            b_cap = 0;
            b_s2_tot = 0; b_s2_pre = 0; b_s2_inf = 0; b_s2_post = 0;
            b_s3_tot = 0; b_s3_pre = 0; b_s3_inf = 0; b_s3_post = 0;
        }

        std::string fps_label = "Pipeline FPS: " + cv::format("%.1f", current_fps);
        cv::putText(p->model_input, fps_label, cv::Point(15, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Active TensorRT 10 Framework Output", p->model_input);
        
        if (cv::waitKey(1) == 27) { 
            running = false;
            break;
        }
    }
}

int runner::setup(int mode, int camera_mode) {
	bool rebuild = false;
	if (mode == 2){
        multicam = false;
		rebuild = true;
	}
    if (camera_mode == 1){
        multicam = false;
        cap1.open(gst_cam1, cv::CAP_GSTREAMER);
        if (!cap1.isOpened()) {
            std::cerr << "Error: Failed to open camera 1 with GStreamer!" << std::endl;
            return -1;
        }
    }
    else if (camera_mode == 2){
        cap1.open(gst_cam2, cv::CAP_GSTREAMER);
        if (!cap1.isOpened()) {
            std::cerr << "Error: Failed to open camera 2 with GStreamer!" << std::endl;
            return -1;
        }
    }
    else if (camera_mode == 3){
        multicam = true;
        cap1.open(gst_cam1, cv::CAP_GSTREAMER);
        cap2.open(gst_cam2, cv::CAP_GSTREAMER);
        bool cap1_test = cap1.isOpened();
        bool cap2_test = cap2.isOpened();
        if (!cap1_test && !cap2_test) {
            std::cerr << "Error: Failed to open camera 1 & camera 2 with GStreamer!" << std::endl;
            return -1;
        }
        else if (!cap1_test && cap2_test){
            std::cerr << "Error: Failed to open camera 1 with GStreamer!" << std::endl;
            return -1;
        } 
        else if (cap1_test && !cap2_test){
            std::cerr << "Error: Failed to open camera 2 with GStreamer!" << std::endl;
            return -1;
        } 
    }
    else {
        std::cerr << "Error: invalid camera mode" << std::endl;
        return -1;
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
		
    bb_ctx_ptr = bbox_runner.getContextPtr();
    bb_cfg_ptr = bbox_runner.getConfigPtr();

	p_logger.initPoseLogger(text_log_file);
	
	return 1;	
}

int runner::run(int mode, int camera_mode) {
    if(setup(mode, camera_mode) != 1){
        std::cout << "runner setup failed" << std::endl;
        return -1;
    }

    std::cout << "Starting 4-Stage Asynchronous Pipeline..." << std::endl;
    running = true;

    global_pool.initialize(10);
    std::cout << "initialize threads\n";
    // Spawn 3 pipeline threads
    std::thread t1(&runner::stage1_capture, this);
    std::thread t2(&runner::stage2_bbox, this);
    std::thread t3(&runner::stage3_pose, this);

    // don't need to thread main
    stage4_output();

    // Wait for shutdown
    t1.join();
    t2.join();
    t3.join();
    
    std::cout << "Pipeline shut down gracefully." << std::endl;
    return 0;
}

runner::runner() : pose_runner(geo) {
}
