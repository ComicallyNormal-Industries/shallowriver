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
