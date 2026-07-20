#include "text_logger.hpp"

void pose_logger::initPoseLogger(std::string& filename){
	poseFile.open(filename, std::ios::out | std::ios::trunc);
	if (!poseFile.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing!" << std::endl;
    }
}

void pose_logger::deinitPoseLogger(){

	if (poseFile.is_open()) {
        poseFile.close();
        std::cout << "Pose log file successfully closed." << std::endl;
    } else {
		
		std::cout << "Problem: deinit called but Pose log file is not open" << std::endl;
	
	}

}

void pose_logger::log_keypoints(const std::vector<NvAR_Point3f>& keypoints){

	if (!poseFile.is_open()) {
        std::cerr << "Warning: Attempted to log keypoints, but pose file is not open!" << std::endl;
        return;
    }

    poseFile << "--- Frame Start ---" << std::endl;
    
    for (size_t i = 0; i < keypoints.size(); ++i) {
        poseFile << "Keypoint_" << i << ": " 
                 << keypoints[i].x << ", " 
                 << keypoints[i].y << ", " 
                 << keypoints[i].z << std::endl;
    }

}

pose_logger::pose_logger() {
    ;
}

// The Destructor (automatically cleans up when the program closes)
pose_logger::~pose_logger() {
    deinitPoseLogger();
}
