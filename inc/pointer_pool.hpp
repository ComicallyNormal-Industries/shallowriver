// global pool of pointers passed through the pipeline stages

using PacketPtr = std::shared_ptr<bb_context_packet>;

struct PacketPool {
    std::queue<bb_context_packet*> free_list;
    std::mutex mtx;

    // Pre-allocate the memory blocks once at startup
    void initialize(int pool_size) {
        for (int i = 0; i < pool_size; ++i) {
            free_list.push(new bb_context_packet());
        }
    }

    bb_context_packet* acquire() {
        std::lock_guard<std::mutex> lock(mtx);
        if (free_list.empty()) {
            std::cerr << "WARNING: Pool empty, pipeline is bottlenecked!" << std::endl;
            return nullptr; 
        }
        bb_context_packet* pkt = free_list.front();
        free_list.pop();
        return pkt;
    }

    void release(bb_context_packet* pkt) {
        std::lock_guard<std::mutex> lock(mtx);
        free_list.push(pkt);
    }
};

// global pool variable. used to get a new context pointer, automatically frees when not used
extern PacketPool global_pool;

// Add 'inline' to the function so the linker knows to merge duplicates
inline PacketPtr get_pooled_packet() {
    bb_context_packet* raw = global_pool.acquire();
    if (!raw) return nullptr;
    
    return std::shared_ptr<bb_context_packet>(raw, [](bb_context_packet* p) {
        global_pool.release(p);
    });
}