#include "intrinsics.hpp"
#include "camera_defs.hpp"
#include "paths.hpp"
// Explicitly define your verified, exact physical board specifications
const int SQUARES_X = 4;               
const int SQUARES_Y = 6;               
const float SQUARE_LENGTH = 40.0f;     
const float MARKER_LENGTH = 30.0f;     
const auto ARUCO_DICT = cv::aruco::DICT_6X6_250; 

int intrinsics::save_intrinsics(int camera_id, cv::Mat cameraMatrix, cv::Mat distCoeffs){
    std::string file_name;
    if (camera_id == 1){
        file_name = paths::data_dir() + "/calibration_1.yaml";
    }
    else if (camera_id == 2){
        file_name = paths::data_dir() + "/calibration_2.yaml";
    }
    else {
        std::cerr << "Error: invalide camera id number cannot save intrinsics to file" << std::endl;
        return -1;
    }

    //Open the file for writing after all safety checks pass
    cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "\n[Error] Failed to open " << file_name << " for writing! Check folder permissions." << std::endl;
        return -1;
    }
    
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs.release();
    return 1;
}

int intrinsics::run_calibration(int camera_id) {
    cv::VideoCapture cap;
    if (camera_id == 1){
        cap.open(gst_calib1, cv::CAP_GSTREAMER);
    }
    else if (camera_id == 2){
        cap.open(gst_calib2, cv::CAP_GSTREAMER);
    }
    else{
        std::cerr << "Error: invalide camera id number" << std::endl;
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Camera device offline." << std::endl;
        return -1;
    }

    // Initialize ChArUco configurations (OpenCV 4 API)
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(ARUCO_DICT);
    cv::aruco::CharucoBoard board(cv::Size(SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary);
    cv::aruco::CharucoDetector detector(board);

    // Extract OpenCV's internally generated, perfectly mapped 3D coordinates
    std::vector<cv::Point3f> board3DPoints = board.getChessboardCorners();

    std::vector<std::vector<cv::Point2f>> allImagePoints;
    std::vector<std::vector<cv::Point3f>> allObjectPoints;
    cv::Size imageSize;

    std::cout << "========================================================\n";
    std::cout << "ChArUco Camera Calibration Active (Fixed Point Mapping)\n";
    std::cout << "-> Press [SPACEBAR] to capture a snapshot view.\n";
    std::cout << "-> Press [ENTER] to execute calculations and save file.\n";
    std::cout << "-> Press [ESC] to cancel.\n";
    std::cout << "========================================================\n" << std::endl;

    cv::Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        imageSize = frame.size();
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect board parameters natively
        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;
        detector.detectBoard(gray, charucoCorners, charucoIds);

        cv::Mat visualizationFrame = frame.clone();
        if (!charucoIds.empty()) {
            cv::aruco::drawDetectedCornersCharuco(visualizationFrame, charucoCorners, charucoIds, cv::Scalar(0, 0, 255));
        }

        std::string statusText = "Saved Viewpoints: " + std::to_string(allImagePoints.size());
        cv::putText(visualizationFrame, statusText, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::imshow("ChArUco Frame Feed", visualizationFrame);

        char key = (char)cv::waitKey(1);
        if (key == 27) return 0; // ESC
        
        if (key == ' ') { // Spacebar to capture
            if (charucoCorners.size() < 4) {
                std::cout << "Frame skipped: Visible corners must be greater than 4." << std::endl;
                continue;
            }

            std::vector<cv::Point2f> validImgPts;
            std::vector<cv::Point3f> validObjPts;

            // Direct index lookup: Completely eliminates orientation/scrambling bugs
            for (size_t i = 0; i < charucoIds.size(); ++i) {
                int id = charucoIds[i];
                
                validImgPts.push_back(charucoCorners[i]);
                validObjPts.push_back(board3DPoints[id]); 
            }

            allImagePoints.push_back(validImgPts);
            allObjectPoints.push_back(validObjPts);
            std::cout << "Stored point layout #" << allImagePoints.size() << " successfully!" << std::endl;
        } 
        else if (key == 13) { // ENTER
            if (allImagePoints.size() < 12) {
                std::cout << "Error: Gather at least 12 highly diverse spatial viewpoints first." << std::endl;
                continue;
            }
            break;
        }
    }

	//Verify data was captured
    if (allImagePoints.empty() || allObjectPoints.empty()) {
        std::cerr << "\n[Warning] No calibration data was collected." << std::endl;
        std::cerr << "Aborting calculation. The existing calibration.yaml will NOT be overwritten." << std::endl;
        return -1;
    }

    std::cout << "\nOptimizing calibration matrix arrays..." << std::endl;
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    double err = -1.0;

    //Catch OpenCV math crashes
    try {
        err = cv::calibrateCamera(allObjectPoints, allImagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    } 
    catch (const cv::Exception& e) {
        std::cerr << "\n[Error] OpenCV Calibration mathematically failed: " << e.what() << std::endl;
        std::cerr << "The existing calibration.yaml will NOT be overwritten." << std::endl;
        return -1;
    }

    //Verify the output matrices are actually valid
    if (cameraMatrix.empty() || err < 0.0) {
        std::cerr << "\n[Error] Invalid camera matrix generated." << std::endl;
        std::cerr << "The existing calibration.yaml will NOT be overwritten." << std::endl;
        return -1;
    }

    std::cout << "\n========================================================" << std::endl;
    std::cout << "Success! Reprojection Error: " << err << " pixels" << std::endl;
    std::cout << "Overwriting calibration.yaml with new data..." << std::endl;
    std::cout << "========================================================" << std::endl;

    int save_err = save_intrinsics(camera_id, cameraMatrix, distCoeffs);

    if (save_err != 1){
        std::cerr << "\n[Error] Saving calibration failed " << std::endl;
    }
    return 0;
}

int intrinsics::generate_charuco() {
    std::cout << "Generating ChArUco board image..." << std::endl;

    //Pull the same dictionary and board specs used for calibration
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(ARUCO_DICT);
    cv::aruco::CharucoBoard board(cv::Size(SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary);

    //Set output image resolution (scaling by 200 pixels per square for high print quality)
    //cv::Size imageSize(SQUARES_X * 200, SQUARES_Y * 200);
    cv::Size imageSize(2000,3000);
	cv::Mat boardImage;

    //Render the board (Image Size, Output Mat, Margin Size, Border Bits)
    board.generateImage(imageSize, boardImage, 0, 1);

    //Save to disk
    std::string board_file = paths::data_dir() + "/charuco_board.png";
    if (cv::imwrite(board_file, boardImage)) {
        std::cout << "Success! Saved '" << board_file << "'." << std::endl;
        std::cout << "Please print this without scaling (100% scale) for accurate calibration." << std::endl;
		std::cout << "Use the following size settings to print the image correctly: " << std::endl;
		std::cout << "Target Width: 6.30 inches (exactly 160 mm)" << std::endl;
		std::cout << "Target Height: 9.45 inches (exactly 240 mm)" << std::endl;
		std::cout << "Measure the arduco markers to make sure they are 30mm x 30mm or the calibration wont work correctly " << std::endl;
		std::cout << "It may be necessary to glue/tape the printed paper to a more rigid surface like cardbaord to keep the charuco board form deforming " << std::endl;
        return 0;
    } else {
        std::cerr << "Error: Failed to write 'charuco_board.png'. Check folder permissions." << std::endl;
        return -1;
    }
}
