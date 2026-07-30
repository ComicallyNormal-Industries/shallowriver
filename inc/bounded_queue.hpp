// inc/bounded_queue.hpp
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

template <typename T>
class BoundedQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
    size_t max_capacity_;
    std::atomic<bool> stopped_{false};

public:
    explicit BoundedQueue(size_t capacity = 3) : max_capacity_(capacity) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_push_.wait(lock, [this]() { return queue_.size() < max_capacity_ || stopped_; });
        if (stopped_) return false;
        queue_.push(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_pop_.wait(lock, [this]() { return !queue_.empty() || stopped_; });
        if (stopped_ && queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return true;
    }

    void stop() {
        stopped_ = true;
        cv_push_.notify_all();
        cv_pop_.notify_all();
    }
};

// Single-producer single-consumer channel that only keeps the latest value.
// T must be default-constructible.
template <typename T>
class SPSCLatestValue {
public:
    SPSCLatestValue() {
        buffers_[0] = std::make_unique<T>();
        buffers_[1] = std::make_unique<T>();
        buffers_[2] = std::make_unique<T>();
        write_buf_ = buffers_[0].get();
        read_buf_ = buffers_[2].get();
        ready_.store(buffers_[1].get(), std::memory_order_relaxed);
    }

    // Non-copyable, non-movable
    SPSCLatestValue(const SPSCLatestValue&) = delete;
    SPSCLatestValue& operator=(const SPSCLatestValue&) = delete;
    SPSCLatestValue(SPSCLatestValue&&) = delete;
    SPSCLatestValue& operator=(SPSCLatestValue&&) = delete;

    template <typename F>
    void produce_update(F&& updater) {
        // Let the caller modify the existing write_buf_ directly
        updater(*write_buf_);
        
        write_buf_ = ready_.exchange(write_buf_, std::memory_order_acq_rel);
        has_new_.store(true, std::memory_order_release);
        has_new_.notify_one();
    }

    // 1. Producer needs to issue the wakeup notification
    template <typename... Args>
    void produce(Args&&... args) {
        *write_buf_ = T(std::forward<Args>(args)...);
        write_buf_ = ready_.exchange(write_buf_, std::memory_order_acq_rel);
        has_new_.store(true, std::memory_order_release);
        
        // Wakes up the consumer if it is currently waiting in has_new_.wait()
        has_new_.notify_one(); 
    }

    // 2. Consumer blocks until notified (or spurious wakeup)
    T* wait_and_consume() {
        while (!has_new_.load(std::memory_order_acquire)) {
            has_new_.wait(false, std::memory_order_relaxed);
        }
        return consume();
    }
    
    // // Producer: write new value, publish it
    // template <typename... Args>
    // void produce(Args&&... args) {
    //     *write_buf_ = T(std::forward<Args>(args)...);
    //     write_buf_ = ready_.exchange(write_buf_, std::memory_order_acq_rel);
    //     has_new_.store(true, std::memory_order_release);
    // }

    // Consumer: get latest value (returns nullptr if nothing new)
    T* consume() {
        if (!has_new_.exchange(false, std::memory_order_acq_rel)) {
            return nullptr;
        }
        T* latest = ready_.exchange(read_buf_, std::memory_order_acq_rel);
        read_buf_ = latest;
        return latest;
    }

    // Consumer: get current read buffer (always valid, may be stale)
    // Only safe to call from consumer thread
    T* current() const { return read_buf_; }

private:
    static constexpr std::size_t cache_line_size = 64;

    std::unique_ptr<T> buffers_[3];

    alignas(cache_line_size) std::atomic<T*> ready_{nullptr};
    alignas(cache_line_size) std::atomic<bool> has_new_{false};

    T* write_buf_ = nullptr;
    T* read_buf_ = nullptr;
};

#include <atomic>
#include <memory>
#include <utility>
#include <new>            // Required for placement new
#include <cuda_runtime.h> // Required for CUDA memory functions

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
    // SPSCLatestValueCuda() {
    //     for (int i = 0; i < 3; ++i) {
    //         T* raw_ptr = nullptr;
            
    //         // Allocate unified memory accessible by both CPU and GPU
    //         cudaMallocManaged(&raw_ptr, sizeof(T));
            
    //         // Initialize the memory by calling T's constructor in-place
    //         new (raw_ptr) T();
            
    //         // Wrap in unique_ptr with our custom CUDA deleter
    //         buffers_[i].reset(raw_ptr);
    //     }

    //     write_buf_ = buffers_[0].get();
    //     read_buf_ = buffers_[2].get();
    //     ready_.store(buffers_[1].get(), std::memory_order_relaxed);
    // }
    SPSCLatestValueCuda() {
        for (int i = 0; i < 3; ++i) {
            // The standard heap allocates the vectors, and the bb_context_packet 
            // constructor automatically handles the CUDA pointers internally!
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

    // Updated array to use the custom CudaDeleter
    // std::unique_ptr<T, CudaDeleter<T>> buffers_[3];
    std::unique_ptr<T> buffers_[3];

    alignas(cache_line_size) std::atomic<T*> ready_{nullptr};
    alignas(cache_line_size) std::atomic<bool> has_new_{false};

    T* write_buf_ = nullptr;
    T* read_buf_ = nullptr;
};
