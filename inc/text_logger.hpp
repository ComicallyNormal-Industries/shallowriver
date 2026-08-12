#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

struct NvAR_Point3f {
    float x, y, z;
};

struct pose_logger {
	std::ofstream poseFile;

	void initPoseLogger(std::string& filename);

	void log_keypoints(const std::vector<NvAR_Point3f>& keypoints);

	void deinitPoseLogger();

	~pose_logger();
};

// Per-frame FPS/latency benchmarking, in the same CSV format as the
// deepstream_pose_estimation_app.cpp --save-perf logger it's meant to be
// directly comparable against:
//   camera_id,frame_number,fps,latency_ms,avg_fps,avg_latency_ms,peak_latency_ms
// `fps` is recomputed once per second from the frames processed in that
// window (matches the deepstream app's windowed calculation instead of an
// instantaneous per-frame value, which would be too noisy to read).
// `avg_fps` is cumulative frame_number / elapsed seconds since the first
// logged frame. latency_ms is supplied by the caller per frame (however this
// pipeline defines "time since the frame entered the system" -- unlike
// DeepStream there's no nvstreammux ntp_timestamp to read here), and
// avg_latency_ms/peak_latency_ms are running stats over all samples seen.
// Disabled (is_enabled() == false) until init() is called with enable=true,
// at which point every method is a no-op -- safe to call unconditionally
// from the pipeline without every call site needing its own enabled check.
struct perf_logger {

	std::ofstream perfFile;
	bool enabled = false;

	std::string camera_id;

	long frame_number = 0;

	std::chrono::steady_clock::time_point start_time{};
	std::chrono::steady_clock::time_point last_fps_window_time{};
	long last_fps_window_frame = 0;
	double current_fps = 0.0;

	double total_latency_ms = 0.0;
	long latency_sample_count = 0;
	double peak_latency_ms = 0.0;


	void init(const std::string& filename, const std::string& camera_id, bool enable);

	// Call once per frame. latency_ms is however the caller measures
	// "time since this frame entered the pipeline" -- this class just
	// tracks/writes the running stats around whatever value it's given.
	void log_frame(double latency_ms);

	void deinit();

	bool is_enabled() const { return enabled; }

	~perf_logger();
};
