
#include <chrono>

// main pipeline context packet passed through the stages

constexpr size_t PEOPLENET_WIDTH = 960;  // 60 grid_w * 16 stride_x
constexpr size_t PEOPLENET_HEIGHT = 544; // 34 grid_h * 16 stride_y
constexpr size_t PEOPLENET_CHANNELS = 3;

constexpr size_t GRID_W = 60;
constexpr size_t GRID_H = 34;
constexpr size_t NUM_CLASSES = 3;

// element counts for the fixed-size arrays
constexpr size_t INPUT_ELEMENTS = PEOPLENET_CHANNELS * PEOPLENET_HEIGHT * PEOPLENET_WIDTH;
constexpr size_t BBOX_ELEMENTS = (NUM_CLASSES * 4) * GRID_H * GRID_W;
constexpr size_t COV_ELEMENTS = NUM_CLASSES * GRID_H * GRID_W;

constexpr size_t FRAME_X = 1920;
constexpr size_t FRAME_Y = 1080;

constexpr size_t TRT_PADDING = 1024;


constexpr int input_w = 192;
constexpr int input_h = 256;
constexpr int num_keypoints = 34;

// NVIDIA standard limb proportions
constexpr std::array<float, 36> scale_normalized_mean_limb_lengths = {
    0.5000f, 0.5000f, 1.0000f, 0.8175f, 0.9889f, 0.2610f, 0.7942f, 0.5724f, 0.5078f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3433f, 0.8171f,
    0.9912f, 0.2610f, 0.8259f, 0.5724f, 0.5078f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.3422f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f
};

constexpr std::array<float, 36> mean_limb_lengths = {
    246.3427f, 246.3427f, 492.6854f, 402.4380f, 487.0321f, 128.6856f, 391.6295f,
    281.9928f, 249.9478f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f,   0.0000f, 169.1832f, 402.2611f, 488.1824f, 128.6848f, 407.5836f,
    281.9897f, 249.9489f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f,   0.0000f, 168.6137f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f
};

struct bb_context_packet
{
    //time stamp variables
    double t_cap = 0.0;

    // Wall-clock moment this frame became available from the camera -- the
    // shallowriver equivalent of DeepStream's nvstreammux ntp_timestamp. Used to
    // compute end-to-end per-frame latency (capture -> output), including any time
    // spent waiting in the inter-stage queues, not just active processing time.
    std::chrono::steady_clock::time_point t_capture_ts{};

    // timestamping varuables
    double t_s2_total = 0.0;
    double t_s2_pre = 0.0;
    double t_s2_inf = 0.0;
    double t_s2_post = 0.0;

    double t_s3_total = 0.0;
    double t_s3_pre = 0.0;
    double t_s3_inf = 0.0;
    double t_s3_post = 0.0;

    // frame counter
    uint64_t frame_id;

    // which camera this frame belongs too
    uint32_t camera_id = 0;

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<int> nms_indices;

    cv::Mat raw_frame;
    cv::Mat model_input;

    // CUDA Pointers 
    float* d_input = nullptr;
    float* d_bbox = nullptr;
    
    float* d_cov = nullptr;

    // Body-pose engine I/O: real device memory (cudaMalloc), not zero-copy mapped host
    // memory. The bodypose3dnet engine's small auxiliary inputs (k_inv, t_form_inv,
    // the limb-length tensors) are not reliably readable by the GPU when bound to a
    // host-mapped pointer on this TensorRT build, even though the pointer is numerically
    // valid under UVA -- it silently produces NaN in the pose3d output. Real device
    // memory plus explicit cudaMemcpy is the pattern confirmed to work.
    float *d_input0 = nullptr, *d_k_inv = nullptr, *d_t_form_inv = nullptr;
    float *d_scale_norm_limb = nullptr, *d_mean_limb = nullptr;

    float *d_pose2d = nullptr, *d_pose2d_org = nullptr, *d_pose25d = nullptr, *d_pose3d = nullptr;

    // Host-side staging/readback buffers paired with the device buffers above.
    // These must be pinned (cudaMallocHost), not plain heap/stack memory: cudaMemcpyAsync
    // against pageable memory silently falls back to a blocking copy, which would defeat
    // the point of putting them on bp_ctx.stream alongside the input0 transfer and inference.
    float* h_input0 = nullptr;
    float* h_k_inv = nullptr;
    float* h_t_form_inv = nullptr;
    float* h_pose2d = nullptr;
    float* h_pose25d = nullptr;
    float* h_pose3d = nullptr;

    // Backing allocations for the k_inv/t_form_inv pair and the pose2d/pose25d/pose3d
    // trio. d_k_inv/d_t_form_inv and h_k_inv/h_t_form_inv above are just offsets into
    // these, so one contiguous cudaMemcpyAsync can move both instead of two separate
    // calls (same idea for the three pose outputs). pose2d_org_img is unused by the
    // rest of the pipeline so it stays a standalone allocation, not worth folding in.
    float* d_k_inv_t_form = nullptr;
    float* h_k_inv_t_form = nullptr;
    float* d_pose_out = nullptr;
    float* h_pose_out = nullptr;

    bb_context_packet() {
        cudaHostAlloc((void**)&d_input, (INPUT_ELEMENTS + TRT_PADDING) * sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc((void**)&d_bbox, (BBOX_ELEMENTS + TRT_PADDING) * sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc((void**)&d_cov, (COV_ELEMENTS + TRT_PADDING) * sizeof(float), cudaHostAllocMapped);

        cudaMalloc((void**)&d_input0, 1 * 3 * input_h * input_w * sizeof(float));
        cudaMalloc((void**)&d_scale_norm_limb, 1 * 36 * sizeof(float));
        cudaMalloc((void**)&d_mean_limb, 1 * 36 * sizeof(float));
        cudaMalloc((void**)&d_pose2d_org, 1 * num_keypoints * 3 * sizeof(float));

        cudaMalloc((void**)&d_k_inv_t_form, (9 + 9) * sizeof(float));
        d_k_inv = d_k_inv_t_form;
        d_t_form_inv = d_k_inv_t_form + 9;

        cudaMalloc((void**)&d_pose_out, (num_keypoints * 3 + num_keypoints * 4 + num_keypoints * 3) * sizeof(float));
        d_pose2d = d_pose_out;
        d_pose25d = d_pose_out + num_keypoints * 3;
        d_pose3d = d_pose_out + num_keypoints * 3 + num_keypoints * 4;

        cudaMallocHost((void**)&h_input0, 1 * 3 * input_h * input_w * sizeof(float));

        cudaMallocHost((void**)&h_k_inv_t_form, (9 + 9) * sizeof(float));
        h_k_inv = h_k_inv_t_form;
        h_t_form_inv = h_k_inv_t_form + 9;

        cudaMallocHost((void**)&h_pose_out, (num_keypoints * 3 + num_keypoints * 4 + num_keypoints * 3) * sizeof(float));
        h_pose2d = h_pose_out;
        h_pose25d = h_pose_out + num_keypoints * 3;
        h_pose3d = h_pose_out + num_keypoints * 3 + num_keypoints * 4;

        std::memset(d_input, 0, (INPUT_ELEMENTS + TRT_PADDING) * sizeof(float));
        std::memset(d_bbox, 0, (BBOX_ELEMENTS + TRT_PADDING) * sizeof(float));
        std::memset(d_cov, 0, (COV_ELEMENTS + TRT_PADDING) * sizeof(float));

        cudaMemset(d_pose2d_org, 0, 1 * num_keypoints * 3 * sizeof(float));
        cudaMemset(d_pose_out, 0, (num_keypoints * 3 + num_keypoints * 4 + num_keypoints * 3) * sizeof(float));
        std::memset(h_pose_out, 0, (num_keypoints * 3 + num_keypoints * 4 + num_keypoints * 3) * sizeof(float));

        cudaMemcpy(d_scale_norm_limb, scale_normalized_mean_limb_lengths.data(), 36 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mean_limb, mean_limb_lengths.data(), 36 * sizeof(float), cudaMemcpyHostToDevice);

        model_input.create(cv::Size(960, 544), CV_8UC3);
    }

    // clean up CUDA memory safely when the packet is destroyed
    ~bb_context_packet() {
if (d_input) cudaFreeHost(d_input);
        if (d_bbox) cudaFreeHost(d_bbox);
        if (d_cov) cudaFreeHost(d_cov);

        if (d_input0) cudaFree(d_input0);
        if (d_k_inv_t_form) cudaFree(d_k_inv_t_form);
        if (d_scale_norm_limb) cudaFree(d_scale_norm_limb);
        if (d_mean_limb) cudaFree(d_mean_limb);

        if (d_pose2d_org) cudaFree(d_pose2d_org);
        if (d_pose_out) cudaFree(d_pose_out);

        if (h_input0) cudaFreeHost(h_input0);
        if (h_k_inv_t_form) cudaFreeHost(h_k_inv_t_form);
        if (h_pose_out) cudaFreeHost(h_pose_out);
    }

    // prevent accidental deep copies that would cause double-frees
    bb_context_packet(const bb_context_packet&) = delete;
    bb_context_packet& operator=(const bb_context_packet&) = delete;
};
