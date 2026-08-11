#include "text_logger.hpp"
#include <iomanip>
#include <algorithm>

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

    poseFile << "--- Frame Start ---\n";

    for (size_t i = 0; i < keypoints.size(); ++i) {
        poseFile << "Keypoint_" << i << ": "
                 << keypoints[i].x << ", "
                 << keypoints[i].y << ", "
                 << keypoints[i].z << '\n';
    }

}

pose_logger::pose_logger() {
    ;
}

pose_logger::~pose_logger() {
    deinitPoseLogger();
}

void perf_logger::init(const std::string& filename, const std::string& camera_id_in, bool enable) {
    enabled = enable;
    if (!enabled) return;

    camera_id = camera_id_in;

    perfFile.open(filename, std::ios::out | std::ios::trunc);
    if (!perfFile.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing!" << std::endl;
        enabled = false;
        return;
    }

    perfFile << "camera_id,frame_number,fps,latency_ms,avg_fps,avg_latency_ms,peak_latency_ms\n";
    perfFile.flush();

    frame_number = 0;
    start_time = std::chrono::steady_clock::time_point{};
    last_fps_window_time = std::chrono::steady_clock::time_point{};
    last_fps_window_frame = 0;
    current_fps = 0.0;
    total_latency_ms = 0.0;
    latency_sample_count = 0;
    peak_latency_ms = 0.0;
}

void perf_logger::log_frame(double latency_ms) {
    if (!enabled) return;

    auto now = std::chrono::steady_clock::now();

    if (frame_number == 0) {
        start_time = now;
        last_fps_window_time = now;
        last_fps_window_frame = 0;
    }

    frame_number++;

    // Recompute the windowed fps once per second, same as the deepstream app --
    // an instantaneous per-frame value would be too noisy to read.
    double window_elapsed_sec = std::chrono::duration<double>(now - last_fps_window_time).count();
    if (window_elapsed_sec >= 1.0) {
        current_fps = (frame_number - last_fps_window_frame) / window_elapsed_sec;
        last_fps_window_time = now;
        last_fps_window_frame = frame_number;
    }

    double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
    double avg_fps = (elapsed_sec > 0.0) ? (frame_number / elapsed_sec) : 0.0;

    total_latency_ms += latency_ms;
    latency_sample_count++;
    peak_latency_ms = std::max(peak_latency_ms, latency_ms);
    double avg_latency_ms = total_latency_ms / latency_sample_count;

    perfFile << camera_id << ','
             << frame_number << ','
             << std::fixed << std::setprecision(2) << current_fps << ','
             << std::setprecision(3) << latency_ms << ','
             << std::setprecision(2) << avg_fps << ','
             << std::setprecision(3) << avg_latency_ms << ','
             << peak_latency_ms << '\n';
    // At only ~5-15 calls/sec the syscall cost is negligible, and it means a crash
    // or abrupt kill loses at most the current row instead of everything still
    // sitting in the stream buffer since the last natural flush.
    perfFile.flush();
}

void perf_logger::deinit() {
    if (!enabled || !perfFile.is_open()) return;

    double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    double avg_fps = (elapsed_sec > 0.0) ? (frame_number / elapsed_sec) : 0.0;
    double avg_latency_ms = (latency_sample_count > 0) ? (total_latency_ms / latency_sample_count) : 0.0;

    perfFile << "\n# Summary: camera_id=" << camera_id
             << ", frames=" << frame_number
             << ", avg_fps=" << std::fixed << std::setprecision(2) << avg_fps
             << ", avg_latency_ms=" << std::setprecision(3) << avg_latency_ms
             << ", peak_latency_ms=" << peak_latency_ms << '\n';
    perfFile.close();

    std::cout << "Perf summary [" << camera_id << "]: frames=" << frame_number
              << ", avg_fps=" << std::fixed << std::setprecision(2) << avg_fps
              << ", avg_latency_ms=" << std::setprecision(3) << avg_latency_ms
              << ", peak_latency_ms=" << peak_latency_ms << std::endl;

    enabled = false;
}

perf_logger::~perf_logger() {
    deinit();
}
