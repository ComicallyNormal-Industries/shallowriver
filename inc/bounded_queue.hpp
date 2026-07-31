#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <utility>
#include <new>
#include <cuda_runtime.h> // Required for CUDA memory functions
#include "pointer_pool.hpp"

// Custom deleter so std::unique_ptr knows how to clean up CUDA memory
template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const {
        if (ptr != nullptr) {
            ptr->~T();         // Explicitly call the destructor
            cudaFree(ptr);     // Free the CUDA memory
        }
    }
};

template <typename T>
class SPSCLatestValueCuda {
public:
    SPSCLatestValueCuda() {
        for (int i = 0; i < 3; ++i) {
            buffers_[i] = std::make_unique<T>(); 
        }

        write_buf_ = buffers_[0].get();
        read_buf_ = buffers_[2].get();
        ready_.store(buffers_[1].get(), std::memory_order_relaxed);
    }

    // Non-copyable, non-movable
    SPSCLatestValueCuda(const SPSCLatestValueCuda&) = delete;
    SPSCLatestValueCuda& operator=(const SPSCLatestValueCuda&) = delete;
    SPSCLatestValueCuda(SPSCLatestValueCuda&&) = delete;
    SPSCLatestValueCuda& operator=(SPSCLatestValueCuda&&) = delete;

    template <typename F>
    void produce_update(F&& updater) {
        updater(*write_buf_);
        write_buf_ = ready_.exchange(write_buf_, std::memory_order_acq_rel);
        has_new_.store(true, std::memory_order_release);
        has_new_.notify_one();
    }
    
    template <typename... Args>
    void produce(Args&&... args) {
        *write_buf_ = T(std::forward<Args>(args)...);
        write_buf_ = ready_.exchange(write_buf_, std::memory_order_acq_rel);
        has_new_.store(true, std::memory_order_release);
        has_new_.notify_one(); 
    }

    T* wait_and_consume() {
        while (!has_new_.load(std::memory_order_acquire)) {
            has_new_.wait(false, std::memory_order_relaxed);
        }
        return consume();
    }
    
    T* consume() {
        if (!has_new_.exchange(false, std::memory_order_acq_rel)) {
            return nullptr;
        }
        T* latest = ready_.exchange(read_buf_, std::memory_order_acq_rel);
        read_buf_ = latest;
        return latest;
    }

    T* current() const { return read_buf_; }

private:
    static constexpr std::size_t cache_line_size = 64;

    std::unique_ptr<T> buffers_[3];

    alignas(cache_line_size) std::atomic<T*> ready_{nullptr};
    alignas(cache_line_size) std::atomic<bool> has_new_{false};

    T* write_buf_ = nullptr;
    T* read_buf_ = nullptr;
};
