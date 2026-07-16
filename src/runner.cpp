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

cv::Mat runner::preprocessFrame(const cv::Mat& frame, cv::Size target_resolution) {
    cv::Mat model_input;
    cv::resize(frame, model_input, target_resolution);
    return cv::dnn::blobFromImage(model_input, 1.0 / 255.0, target_resolution, cv::Scalar(0,0,0), true, false);
}

void runner::decodeDetections(const TRTContext& trt, const ModelConfig& cfg, std::vector<cv::Rect>& bboxes, std::vector<float>& confidences, std::vector<int>& class_ids) {
    			
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

std::vector<int> runner::applyNMSAndRender(cv::Mat& output_image, const ModelConfig& cfg, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidences, const std::vector<int>& class_ids) {
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

std::vector<NvAR_Point3f> runner::processBodyPoseOutput(
    	const std::vector<float>& pose25d,
    	const std::vector<float>& pose3d_raw,
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
        // 1. Get raw crop pixels from 2.5D output
        float crop_x = pose25d[k * 4 + 0];
        float crop_y = pose25d[k * 4 + 1];

        // 2. Map crop pixels back to full 960x544 frame
        float scale_x = static_cast<float>(person_box.width) / crop_w;
        float scale_y = static_cast<float>(person_box.height) / crop_h;
        float full_x = person_box.x + (crop_x * scale_x);
        float full_y = person_box.y + (crop_y * scale_y);

        // 3. Get the PERFECT Absolute Depth (Z) calculated by the GPU
        float z_abs = pose3d_raw[k * 3 + 2];

        // 4. Standard Pinhole Camera Projection (Pixels -> Metric World Space)
        final_3d[k].x = (full_x - cx) * z_abs / fx;
        final_3d[k].y = (full_y - cy) * z_abs / fy;
        final_3d[k].z = z_abs;
    }

    return final_3d;
}

void runner::setup() {

    cv::VideoCapture cap;
    stream_resolution = cv::Size(1920, 1080);
    peoplenet_resolution = cv::Size(960, 544);
    onnx_file = "resnet34_peoplenet.onnx";
    engine_file = "peoplenet.engine";
    bp_onnx_file = "bodypose3dnet_performance.onnx";
    bp_engine_file = "bodypose3dnet_performance.engine";





	// Load Calibration
    CameraGeometry geo;
    if (!loadAndScaleIntrinsics("calibration.yaml", stream_resolution, peoplenet_resolution, geo)) {
        std::cerr << "Warning: Could not load calibration data." << std::endl;
    }
	cap.open(0);	
    cap.set(cv::CAP_PROP_FRAME_WIDTH, stream_resolution.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, stream_resolution.height);
			
}

int runner::run() {
	setup();

	std::cout << "Starting real-time production TensorRT 10 execution loop..." << std::endl;
    cv::Mat frame, model_input;

	while (cv::waitKey(1) != 27) { // Press ESC to terminate cleanly
        cap >> frame;
        if (frame.empty()) break;

        cv::resize(frame, model_input, peoplenet_resolution);
        cv::Mat input_blob = preprocessFrame(frame, peoplenet_resolution);

        //runInference(trt_ctx, input_blob);
		runInference(input_blob);


        std::vector<cv::Rect> bboxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        //decodeDetections(trt_ctx, config, bboxes, confidences, class_ids);
		decodeDetections(config, bboxes, confidences, class_ids);        


        // Render boxes first
        std::vector<int> nms_indices = applyNMSAndRender(model_input, config, bboxes, confidences, class_ids);
    
        // iterate over indices to define person_box and run keypoints
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
                processAndRunBodyPose(model_input, person_box);
		
				//Get the focal length from your scaled intrinsic matrix
                float focal_length = static_cast<float>(geo.cameraMatrixScaled.at<double>(0, 0));
                
                // Process the coordinates using the true depth and un-cropped pixels
                std::vector<NvAR_Point3f> final3D = processBodyPoseOutput(
                    bp_ctx.h_pose25d, 
                    bp_ctx.h_pose3d, 
                    bp_config.num_keypoints, 
                    person_box,
                    bp_config.input_w,
                    bp_config.input_h,
                    geo.cameraMatrixScaled
                );
				
				/* add log functionality back
                // Log the final metric data
                if (bp_ctx.poseFile.is_open()) {
                    bp_ctx.poseFile << "--- Frame Start ---" << std::endl;
                    for (int k = 0; k < bp_config.num_keypoints; ++k) {
                        bp_ctx.poseFile << "Keypoint_" << k << ": " 
                                        << final3D[k].x << ", " 
                                        << final3D[k].y << ", " 
                                        << final3D[k].z << std::endl;
                    }
                }		
				*/
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
	//add de init

}



