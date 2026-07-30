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

void runner::preprocessFrame(const cv::Mat& frame, cv::Size target_resolution, cv::Mat& blob) {
    // cv::resize(frame, out_model_input, target_resolution);
    // blob = cv::dnn::blobFromImage(out_model_input, 1.0 / 255.0, target_resolution, cv::Scalar(0,0,0), true, false);
    cv::dnn::blobFromImage(
        frame,              // Input image
        blob,               // Output array (writes directly into your pre-allocated CUDA Mat)
        1.0 / 255.0,        // Scale factor
        target_resolution,  // Target size (replaces the need for cv::resize)
        cv::Scalar(0,0,0),  // Mean subtraction
        true,               // SwapRB (BGR to RGB)
        false,              // Crop
        CV_32F              // Depth
    );
}

void runner::decodeDetections(const ModelConfig& cfg, bb_context_packet& bb_context) {

    int stride_spatial = cfg.grid_h * cfg.grid_w;

    for (int c = 0; c < cfg.num_classes; ++c) {
        for (int y = 0; y < cfg.grid_h; ++y) {
            for (int x = 0; x < cfg.grid_w; ++x) {

                int cov_offset = (c * stride_spatial) + (y * cfg.grid_w) + x;
                
                // UPDATED: Reading from the safe CPU vector instead of Unified Memory
                float confidence = bb_context.d_cov[cov_offset];

                if (confidence >= cfg.conf_threshold) {
                    int base_bbox_class = (c * 4);
                    int idx_x1 = ((base_bbox_class + 0) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_y1 = ((base_bbox_class + 1) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_x2 = ((base_bbox_class + 2) * stride_spatial) + (y * cfg.grid_w) + x;
                    int idx_y2 = ((base_bbox_class + 3) * stride_spatial) + (y * cfg.grid_w) + x;

                    // UPDATED: Reading from the safe CPU vector
                    float dx1 = bb_context.d_cov[idx_x1];
                    float dy1 = bb_context.d_cov[idx_y1];
                    float dx2 = bb_context.d_cov[idx_x2];
                    float dy2 = bb_context.d_cov[idx_y2];

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

// --- STAGE 1: Frame Capture ---
void runner::stage1_capture() {
    uint64_t frame_counter = 0;
    while (running) {
        std::cout << "here\n";
        cv::Mat frame;
        cap >> frame; 
        if (frame.empty()) {
            running = false;
            break;
        }
        std::cout << "here1\n";
        // CRITICAL FIX: Clone the frame so the capture thread doesn't overwrite it
        // if (!q1_2.produce({frame_counter++, frame.clone()})) break;
        // q1_2.produce_update([&](FramePacket& data) {
        //     data.frame_id = frame_counter++;
        //     // Reuses the buffer inside data.image if the size/type matches
        //     frame.copyTo(data.raw_frame); 
        // });
        std::cout << "here2\n";
        q1_2.produce_update([&](bb_context_packet& data) {
            std::cout << "here5\n";
            data.frame_id = frame_counter++;
            std::cout << "here3\n";
            frame.copyTo(data.raw_frame); 
        });
        std::cout << "here4\n";
    }
    // q1_2.stop();
}

// --- STAGE 2: Model 1 (PeopleNet Bounding Box) ---
void runner::stage2_bbox() {
    cudaSetDevice(0); // Bind CUDA context
    // FramePacket in;
    // while (q1_2.pop(in)) {
    while (true) {

        bb_context_packet* bb_packet = q1_2.wait_and_consume();         

        // cv::Mat model_input, input_blob;
        preprocessFrame(bb_packet->raw_frame, peoplenet_resolution, bb_packet->model_input);
        
        

        // 1. Inference (cudaMemcpy inside this function will now handle locks safely)
        if (bbox_runner.runInference(*bb_packet))
        {

        }
        else
        {
            break;
        }
        
        
        // 2. Wait for inference to finish writing the output
        // cudaDeviceSynchronize();
        
        // 3. CLONE THE OUTPUTS to safe CPU heap memory
        // int spatial_size = bb_cfg_ptr->grid_h * bb_cfg_ptr->grid_w;
        // int cov_elements = bb_cfg_ptr->num_classes * spatial_size;
        // int bbox_elements = bb_cfg_ptr->num_classes * 4 * spatial_size;

        // std::vector<float> safe_cov(cov_elements);
        // std::vector<float> safe_bbox(bbox_elements);

        // cudaMemcpy(safe_cov.data(), bb_ctx_ptr->d_cov, cov_elements * sizeof(float), cudaMemcpyDefault);
        // cudaMemcpy(safe_bbox.data(), bb_ctx_ptr->d_bbox, bbox_elements * sizeof(float), cudaMemcpyDefault);
        
        // 4. Postprocess using the safe CPU clones (You will need to update decodeDetections to accept these vectors)
        // std::vector<cv::Rect> bboxes;
        // std::vector<float> confidences;
        // std::vector<int> class_ids;

        decodeDetections(*bb_cfg_ptr, *bb_packet);
        auto nms_indices = applyNMS(*bb_cfg_ptr, bb_packet->bboxes, bb_packet->confidences);
        
        renderDetections(bb_packet->model_input, *bb_cfg_ptr, *bb_packet, nms_indices);
        
        // q1_2.produce_update([&](bb_context_packet& data) {
        //     data.frame_id = frame_counter++;
        //     // Reuses the buffer inside data.image if the size/type matches
        //     bb_packet.raw_frame.copyTo(data.raw_frame); 
        //     bb_packet.blob_input.copyTo(data.blob_input);


        // });
        q2_3.produce_update([&](bb_context_packet& data) {

            // 1. ZERO-COPY VECTORS: std::move transfers the internal memory pointers 
            // from bb_packet directly to the queue's data without copying any elements.
            data.bboxes = std::move(bb_packet->bboxes);
            data.confidences = std::move(bb_packet->confidences);
            data.class_ids = std::move(bb_packet->class_ids);
            data.nms_indices = std::move(bb_packet->nms_indices);

            // 2. ZERO-COPY CPU MAT: std::move transfers the OpenCV header and reference count.
            // It points data.raw_frame to the exact same pixel memory as bb_packet.raw_frame.
            data.raw_frame = std::move(bb_packet->raw_frame);

            // 3. CUDA-MAPPED MAT: You CANNOT use std::move here.
            // Because data.model_input is mapped to your fixed unified memory array (d_input), 
            // std::move would destroy that mapping. You must use copyTo() so it writes 
            // the underlying float values directly into the CUDA d_input array.
            if (!bb_packet->model_input.empty()) {
                bb_packet->model_input.copyTo(data.model_input);
            }
        });
        // if (!q2_3.push({in.frame_id, in.raw_frame, model_input, bboxes, confidences, class_ids, nms_indices})) break;
    }
    // q2_3.stop();
}

// --- STAGE 3: Model 2 (Body Pose 3D) ---
void runner::stage3_pose() {
    cudaSetDevice(0); // Bind CUDA context
    bb_context_packet* in;
    // while (q2_3.pop(in)) {
    while (true) {

        in = q2_3.wait_and_consume();

        // --- VIEW THE FRAME AT THE START OF STAGE 3 ---

        cv::imshow("Stage 3 Input", in->model_input);
        cv::waitKey(1); // Refresh the GUI window (1ms delay)
        
        continue;

        // Loop through all detected people
        for (int idx : in->nms_indices) {
            if (in->class_ids[idx] == 0) { 
                cv::Rect box = in->bboxes[idx];
                box.x = std::max(0, box.x - 10);
                box.y = std::max(0, box.y - 10);
                box.width = std::min(in->model_input.cols - box.x, box.width + 20);
                box.height = std::min(in->model_input.rows - box.y, box.height + 20);

                cv::Mat crop_blob, t_form_inv;
                
                // 1. Preprocess Crop
                preprocessBodyPoseInput(in->model_input, box, bp_cfg_ptr->input_w, bp_cfg_ptr->input_h, crop_blob, t_form_inv);

                // 2. Inference
                pose_runner.run(crop_blob, t_form_inv);

                // 3. Wait for GPU to finish inference
                cudaDeviceSynchronize();

                // 4. CLONE THE OUTPUTS to safe CPU memory
                int num_kpts = bp_cfg_ptr->num_keypoints;
                
                std::vector<float> safe_pose25d(num_kpts * 4);
                std::vector<float> safe_pose3d(num_kpts * 3);
                std::vector<float> safe_pose2d(num_kpts * 3);

                cudaMemcpy(safe_pose25d.data(), bp_ctx_ptr->d_pose25d, num_kpts * 4 * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(safe_pose3d.data(), bp_ctx_ptr->d_pose3d, num_kpts * 3 * sizeof(float), cudaMemcpyDefault);
                cudaMemcpy(safe_pose2d.data(), bp_ctx_ptr->d_pose2d, num_kpts * 3 * sizeof(float), cudaMemcpyDefault);

                // 5. Postprocess using safe CPU vectors
                std::vector<NvAR_Point3f> final_3d = processBodyPoseOutput(
                    safe_pose25d.data(), safe_pose3d.data(),
                    num_kpts, box,
                    bp_cfg_ptr->input_w, bp_cfg_ptr->input_h,
                    pose_runner.geo.cameraMatrixOrig
                );
                p_logger.log_keypoints(final_3d);

                // 6. Draw Skeleton using safe CPU vector
                for (int k = 0; k < num_kpts; ++k) {
                    float kx = safe_pose2d[k * 3 + 0];
                    float ky = safe_pose2d[k * 3 + 1];
                    float conf = safe_pose2d[k * 3 + 2];
                    if (conf > 0.5f) {
                        int ax = static_cast<int>(t_form_inv.at<float>(0, 0) * kx + t_form_inv.at<float>(0, 1) * ky + t_form_inv.at<float>(0, 2));
                        int ay = static_cast<int>(t_form_inv.at<float>(1, 0) * kx + t_form_inv.at<float>(1, 1) * ky + t_form_inv.at<float>(1, 2));
                        cv::circle(in->model_input, cv::Point(ax, ay), 4, cv::Scalar(0, 255, 255), -1);
                    }
                }
            }
        }
        
        // Push to output
        if (!q3_4.push({in->frame_id, in->model_input})) break;
    }
    q3_4.stop();
}

// --- STAGE 4: Output Display ---
void runner::stage4_output() {
    RenderPacket in;
    
    // Setup FPS tracking variables
    auto fps_start_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    double current_fps = 0.0;

    while (q3_4.pop(in)) {
        // Increment frames processed
        frame_count++;
        auto current_time = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(current_time - fps_start_time).count();

        // Update the FPS calculation every 0.5 seconds for readability
        if (elapsed_seconds >= 0.5) {
            current_fps = frame_count / elapsed_seconds;
            fps_start_time = current_time;
            frame_count = 0;
        }

        // Draw the FPS metric onto the frame (Green text, top left)
        std::string fps_label = "Pipeline FPS: " + cv::format("%.1f", current_fps);
        cv::putText(in.final_frame, fps_label, cv::Point(15, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // Display the output
        cv::imshow("Active TensorRT 10 Framework Output", in.final_frame);
        
        if (cv::waitKey(1) == 27) { // ESC key
            running = false;
            
            // Stop queues to unblock sleeping threads
            // q1_2.stop();
            // q2_3.stop();
            q3_4.stop();
            break;
        }
    }
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

int runner::run(int mode) {
    if(!setup(mode)){
        std::cout << "runner setup failed" << std::endl;
        return -1;
    }

    std::cout << "Starting 4-Stage Asynchronous Pipeline..." << std::endl;
    running = true;

    // Spawn 4 pipeline threads
    std::thread t1(&runner::stage1_capture, this);
    std::thread t2(&runner::stage2_bbox, this);
    std::thread t3(&runner::stage3_pose, this);
    std::thread t4(&runner::stage4_output, this);

    // Wait for shutdown
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    std::cout << "Pipeline shut down gracefully." << std::endl;
    return 0;
}

runner::runner() : pose_runner(geo), cap(0, cv::CAP_V4L2) {
    // The body can stay completely empty.
}
