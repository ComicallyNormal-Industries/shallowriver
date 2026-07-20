#include <iostream>
#include <fstream> 
#include <vector>
#include <string>

struct NvAR_Point3f {
    float x, y, z;
};

class pose_logger {
	private:
		std::ofstream poseFile;
		
		
		
	public:
		void initPoseLogger(std::string& filename);

		void log_keypoints(const std::vector<NvAR_Point3f>& keypoints);

		void deinitPoseLogger();

		pose_logger();
    	~pose_logger();	
		
};
