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

void runner::preprocessBodyPoseInput(const cv::Mat& original_frame, const cv::Rect& person_box, int input_w, int input_h, cv::Mat& out_blob, cv::Mat& out_t_form_inv) {
    
    cv::Point2f src_pts[3], dst_pts[3];
    src_pts[0] = cv::Point2f(person_box.x, person_box.y);
    src_pts[1] = cv::Point2f(person_box.x + person_box.width, person_box.y);
    src_pts[2] = cv::Point2f(person_box.x, person_box.y + person_box.height);
    
    dst_pts[0] = cv::Point2f(0, 0);
    dst_pts[1] = cv::Point2f(input_w, 0);
    dst_pts[2] = cv::Point2f(0, input_h);

    cv::Mat t_form = cv::getAffineTransform(src_pts, dst_pts);
    cv::Mat t_form_3x3 = cv::Mat::eye(3, 3, CV_32F);
    t_form.convertTo(t_form_3x3(cv::Rect(0, 0, 3, 2)), CV_32F);
    
    cv::Mat t_form_inv_double = t_form_3x3.inv(); 
    
    // Assign directly to the output reference
    t_form_inv_double.convertTo(out_t_form_inv, CV_32F);
    out_t_form_inv = out_t_form_inv.clone();

    cv::Mat cropped_person;
    cv::warpAffine(original_frame, cropped_person, t_form, cv::Size(input_w, input_h));
    
    // Assign directly to the output reference
    out_blob = cv::dnn::blobFromImage(cropped_person, 1.0/255.0, cv::Size(), cv::Scalar(0,0,0), true, false);
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

int runner::setup() {

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
	//cap.open(0);	
    cap.set(cv::CAP_PROP_FRAME_WIDTH, stream_resolution.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, stream_resolution.height);
	
	std::cout << "set up bounding box runner" << std::endl;
	if(!bbox_runner.setup(bb_engine_file,bb_onnx_file, peoplenet_resolution)){
		std::cout << "set up bounding box runner failed" << std::endl;
		return -1;
	}
	
	std::cout << "set up pose estimation runner" << std::endl;
	if(!pose_runner.setup(bp_engine_file,bp_onnx_file, cv::Size(192, 256), geo)){
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

int runner::run() {
	if(!setup()){
		std::cout << "runner setup failed" << std::endl;
		return -1;
	}

	std::cout << "Starting real-time production TensorRT 10 execution loop..." << std::endl;
    cv::Mat frame, model_input;
	std::cout << "preloop" << std::endl;
	while (cv::waitKey(1) != 27) { // Press ESC to terminate cleanly
        //std::cout << "Start loop " << std::endl;
		cap >> frame;
        if (frame.empty()) { 
			std::cerr << "frame capture failed ... " << std::endl;
			break;
		}

		//std::cout << "frame captured" << std::endl;

        cv::resize(frame, model_input, peoplenet_resolution);
        cv::Mat input_blob = preprocessFrame(frame, model_input, peoplenet_resolution);

		//std::cout << "preprocess frame successful " << std::endl;

		//run inference on bounding box model
		if(!bbox_runner.run(input_blob)){
			std::cout << "running bounding box model failed" << std::endl;
			return -1;
		}

		//std::cout << "bounding box run successful" << std::endl;

		//find bounding boxes from model output
        std::vector<cv::Rect> bboxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
		decodeDetections(*bb_ctx_ptr, *bb_cfg_ptr, bboxes, confidences, class_ids);        

		//std::cout << "decode detections successful" << std::endl;
        
		// Render boxes first
        
		std::vector<int> nms_indices = applyNMSAndRender(model_input, *bb_cfg_ptr, bboxes, confidences, class_ids);
   		
		//std::cout << "nms_indices successful" << std::endl; 
        
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

				//cpu preprocessing				
				cv::Mat blob;
        		cv::Mat t_form_inv;
				preprocessBodyPoseInput(model_input, person_box, bp_cfg_ptr->input_w, bp_cfg_ptr->input_h, blob, t_form_inv);

                // Run inference on the crop
                //processAndRunBodyPose(model_input, person_box);
				if(!pose_runner.run(blob, t_form_inv)){
					std::cout << "runnning inference on body pose failed" << std::endl;
				}		
				//std::cout << "running body pose successfull" << std::endl;	

				//Get the focal length from your scaled intrinsic matrix
                float focal_length = static_cast<float>(geo.cameraMatrixScaled.at<double>(0, 0));
                
                // Process the coordinates using the true depth and un-cropped pixels
				//std::cout << "crop size " << bp_cfg_ptr->input_w << " " << bp_cfg_ptr->input_h << std::endl;
				std::vector<NvAR_Point3f> final3D = processBodyPoseOutput(
                    bp_ctx_ptr->h_pose25d, 
                    bp_ctx_ptr->h_pose3d, 
                    bp_cfg_ptr->num_keypoints, 
                    person_box,
                    bp_cfg_ptr->input_w,
                    bp_cfg_ptr->input_h,
                    geo.cameraMatrixScaled
                );
				//std::cout << "process body pose output successful" << std::endl;	
			
				//log 3d points to text file
				p_logger.log_keypoints(final3D);

	 
                // Draw keypoints inside the person loop
                for (int k = 0; k < bp_cfg_ptr->num_keypoints; ++k) {
                    float kx_crop = bp_ctx_ptr->h_pose2d[k * 3 + 0];
                    float ky_crop = bp_ctx_ptr->h_pose2d[k * 3 + 1];
                    float conf    = bp_ctx_ptr->h_pose2d[k * 3 + 2];
                    
                    if (conf > 0.2f) {
                        int actual_x = person_box.x + static_cast<int>((kx_crop / bp_cfg_ptr->input_w) * person_box.width);
                        int actual_y = person_box.y + static_cast<int>((ky_crop / bp_cfg_ptr->input_h) * person_box.height);
                        
                        cv::circle(model_input, cv::Point(actual_x, actual_y), 4, cv::Scalar(0, 255, 255), -1);
                    }
                }
            }   
		}
		cv::imshow("Active TensorRT 10 Framework Output", model_input);
	}
	//add de init
	return 0;
}


runner::runner() : pose_runner(geo), cap(0) {
    // The body can stay completely empty.
}
