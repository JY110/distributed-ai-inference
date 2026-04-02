#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <atomic>
#include <memory>
#include <chrono>
#include <string>
#include <stdexcept>

namespace distributed_inference {

/**
 * @brief 批处理请求接口
 * 
 * 所有需要批处理的请求都应实现此接口
 */
template <typename Input, typename Output>
class BatchRequest {
 public:
  virtual ~BatchRequest() = default;
  
  /**
   * @brief 获取请求输入数据
   */
  virtual const Input& GetInput() const = 0;
  
  /**
   * @brief 设置请求输出结果
   */
  virtual void SetOutput(const Output& output) = 0;
  
  /**
   * @brief 设置请求错误
   */
  virtual void SetError(const std::string& error) = 0;
  
  /**
   * @brief 检查请求是否已完成
   */
  virtual bool IsCompleted() const = 0;
  
  /**
   * @brief 等待请求完成
   */
  virtual void Wait() = 0;
  
  /**
   * @brief 获取请求创建时间
   */
  virtual std::chrono::steady_clock::time_point GetCreationTime() const = 0;
};

/**
 * @brief 批处理器类
 * 
 * 设计思路：
 * 1. 收集单个请求，形成批处理
 * 2. 当达到批处理大小或超时时间时，执行批处理
 * 3. 支持多线程并行处理
 * 4. 线程安全设计
 */
template <typename Input, typename Output>
class BatchProcessor {
 public:
  struct Config {
    size_t max_batch_size = 32;      // 最大批处理大小
    int batch_timeout_ms = 100;      // 批处理超时时间(毫秒)
    int num_threads = 4;             // 处理线程数
    int queue_capacity = 1000;       // 队列容量
  };
  
  using ProcessFunc = std::function<std::vector<Output>(const std::vector<Input>&)>;
  
  /**
   * @brief 构造函数
   * @param config 批处理器配置
   * @param process_func 批处理函数
   */
  BatchProcessor(const Config& config, ProcessFunc process_func);
  
  /**
   * @brief 析构函数
   */
  ~BatchProcessor();
  
  /**
   * @brief 启动批处理器
   */
  void Start();
  
  /**
   * @brief 停止批处理器
   */
  void Stop();
  
  /**
   * @brief 提交请求
   * @param request 批处理请求
   * @return 是否提交成功
   */
  bool Submit(std::shared_ptr<BatchRequest<Input, Output>> request);
  
  /**
   * @brief 获取当前队列大小
   */
  size_t GetQueueSize() const;
  
  /**
   * @brief 获取批处理统计
   */
  struct Stats {
    int64_t total_batches;         // 总批处理次数
    int64_t total_requests;        // 总请求数
    int64_t avg_batch_size;        // 平均批处理大小
    int64_t total_process_time_ms; // 总处理时间
    int64_t avg_process_time_ms;   // 平均处理时间
  };
  Stats GetStats() const;

 private:
  /**
   * @brief 工作线程函数
   */
  void WorkerThread();
  
  /**
   * @brief 处理批处理
   */
  void ProcessBatch(std::vector<std::shared_ptr<BatchRequest<Input, Output>>> batch);
  
  /**
   * @brief 检查是否需要触发批处理
   */
  bool ShouldTriggerBatch() const;

 private:
  Config config_;
  ProcessFunc process_func_;
  
  std::queue<std::shared_ptr<BatchRequest<Input, Output>>> request_queue_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  
  std::atomic<bool> is_running_{false};
  std::vector<std::thread> worker_threads_;
  
  // 统计信息
  std::atomic<int64_t> total_batches_{0};
  std::atomic<int64_t> total_requests_{0};
  std::atomic<int64_t> total_process_time_ms_{0};
  mutable std::mutex stats_mutex_;
  
  // 批处理时间戳
  std::chrono::steady_clock::time_point last_batch_time_;
};

/**
 * @brief 批处理请求实现
 */
template <typename Input, typename Output>
class SimpleBatchRequest : public BatchRequest<Input, Output> {
 public:
  SimpleBatchRequest(const Input& input) 
      : input_(input), 
        completed_(false),
        creation_time_(std::chrono::steady_clock::now()) {}
  
  const Input& GetInput() const override {
    return input_;
  }
  
  void SetOutput(const Output& output) override {
    std::lock_guard<std::mutex> lock(mutex_);
    output_ = output;
    error_ = "";
    completed_ = true;
    cv_.notify_one();
  }
  
  void SetError(const std::string& error) override {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = error;
    completed_ = true;
    cv_.notify_one();
  }
  
  bool IsCompleted() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
  }
  
  void Wait() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return completed_; });
  }
  
  std::chrono::steady_clock::time_point GetCreationTime() const override {
    return creation_time_;
  }
  
  // 获取输出
  const Output& GetOutput() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!completed_ || !error_.empty()) {
      throw std::runtime_error("Request not completed or has error: " + error_);
    }
    return output_;
  }
  
  // 获取错误
  const std::string& GetError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
  }

 private:
  Input input_;
  Output output_;
  std::string error_;
  bool completed_;
  std::chrono::steady_clock::time_point creation_time_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
};

// ==================== 模板实现 ====================

template <typename Input, typename Output>
BatchProcessor<Input, Output>::BatchProcessor(const Config& config, ProcessFunc process_func)
    : config_(config),
      process_func_(process_func),
      last_batch_time_(std::chrono::steady_clock::now()) {
}

template <typename Input, typename Output>
BatchProcessor<Input, Output>::~BatchProcessor() {
  Stop();
}

template <typename Input, typename Output>
void BatchProcessor<Input, Output>::Start() {
  if (is_running_.load()) {
    return;
  }
  
  is_running_.store(true);
  
  // 启动工作线程
  for (int i = 0; i < config_.num_threads; ++i) {
    worker_threads_.emplace_back([this]() { WorkerThread(); });
  }
}

template <typename Input, typename Output>
void BatchProcessor<Input, Output>::Stop() {
  if (!is_running_.load()) {
    return;
  }
  
  is_running_.store(false);
  queue_cv_.notify_all();
  
  for (auto& thread : worker_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  
  worker_threads_.clear();
}

template <typename Input, typename Output>
bool BatchProcessor<Input, Output>::Submit(std::shared_ptr<BatchRequest<Input, Output>> request) {
  if (!is_running_.load()) {
    return false;
  }
  
  std::lock_guard<std::mutex> lock(queue_mutex_);
  
  if (request_queue_.size() >= config_.queue_capacity) {
    return false; // 队列已满
  }
  
  request_queue_.push(request);
  total_requests_++;
  
  // 检查是否需要立即触发批处理
  if (ShouldTriggerBatch()) {
    queue_cv_.notify_one();
  }
  
  return true;
}

template <typename Input, typename Output>
void BatchProcessor<Input, Output>::WorkerThread() {
  while (is_running_.load()) {
    std::vector<std::shared_ptr<BatchRequest<Input, Output>>> batch;
    
    { 
      std::unique_lock<std::mutex> lock(queue_mutex_);
      
      // 等待条件：队列不为空 或 超时 或 停止
      auto timeout = std::chrono::milliseconds(config_.batch_timeout_ms);
      
      bool triggered = queue_cv_.wait_for(lock, timeout, [this]() {
        return !request_queue_.empty() || !is_running_.load();
      });
      
      if (!is_running_.load()) {
        break;
      }
      
      if (!triggered) {
        // 超时，检查是否有请求需要处理
        if (request_queue_.empty()) {
          continue;
        }
      }
      
      // 收集批处理请求
      size_t batch_size = std::min(config_.max_batch_size, request_queue_.size());
      for (size_t i = 0; i < batch_size && !request_queue_.empty(); ++i) {
        batch.push_back(request_queue_.front());
        request_queue_.pop();
      }
    }
    
    if (!batch.empty()) {
      ProcessBatch(batch);
    }
  }
}

template <typename Input, typename Output>
void BatchProcessor<Input, Output>::ProcessBatch(
    std::vector<std::shared_ptr<BatchRequest<Input, Output>>> batch) {
  auto start_time = std::chrono::steady_clock::now();
  
  try {
    // 收集输入数据
    std::vector<Input> inputs;
    inputs.reserve(batch.size());
    for (const auto& request : batch) {
      inputs.push_back(request->GetInput());
    }
    
    // 执行批处理
    std::vector<Output> outputs = process_func_(inputs);
    
    // 分发结果
    if (outputs.size() == batch.size()) {
      for (size_t i = 0; i < batch.size(); ++i) {
        batch[i]->SetOutput(outputs[i]);
      }
    } else {
      // 结果数量不匹配
      for (const auto& request : batch) {
        request->SetError("Batch processing failed: output size mismatch");
      }
    }
    
  } catch (const std::exception& e) {
    // 处理异常
    for (const auto& request : batch) {
      request->SetError(std::string("Batch processing failed: ") + e.what());
    }
  }
  
  // 更新统计
  auto end_time = std::chrono::steady_clock::now();
  int64_t process_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();
  
  total_batches_++;
  total_process_time_ms_ += process_time_ms;
  last_batch_time_ = end_time;
}

template <typename Input, typename Output>
bool BatchProcessor<Input, Output>::ShouldTriggerBatch() const {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  
  // 达到最大批处理大小
  if (request_queue_.size() >= config_.max_batch_size) {
    return true;
  }
  
  // 检查是否超时
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - last_batch_time_).count();
  
  return elapsed >= config_.batch_timeout_ms && !request_queue_.empty();
}

template <typename Input, typename Output>
size_t BatchProcessor<Input, Output>::GetQueueSize() const {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return request_queue_.size();
}

template <typename Input, typename Output>
typename BatchProcessor<Input, Output>::Stats BatchProcessor<Input, Output>::GetStats() const {
  Stats stats;
  stats.total_batches = total_batches_.load();
  stats.total_requests = total_requests_.load();
  stats.total_process_time_ms = total_process_time_ms_.load();
  
  if (stats.total_batches > 0) {
    stats.avg_batch_size = stats.total_requests / stats.total_batches;
    stats.avg_process_time_ms = stats.total_process_time_ms / stats.total_batches;
  } else {
    stats.avg_batch_size = 0;
    stats.avg_process_time_ms = 0;
  }
  
  return stats;
}

}  // namespace distributed_inference
