#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace distributed_inference {

/**
 * @brief 内存池类
 * 
 * 设计思路：
 * 1. 预分配固定大小的内存块
 * 2. 维护空闲内存块列表
 * 3. 支持内存块的分配和回收
 * 4. 线程安全设计
 * 5. 支持不同大小的内存块池
 */
template <typename T>
class MemoryPool {
 public:
  /**
   * @brief 构造函数
   * @param block_size 单个内存块大小
   * @param block_count 初始内存块数量
   * @param growth_factor 内存不足时的增长因子
   */
  MemoryPool(size_t block_size, size_t block_count = 10, float growth_factor = 1.5f);
  
  /**
   * @brief 析构函数
   */
  ~MemoryPool();
  
  /**
   * @brief 分配内存块
   * @return 分配的内存块指针
   */
  T* Allocate();
  
  /**
   * @brief 回收内存块
   * @param ptr 要回收的内存块指针
   */
  void Deallocate(T* ptr);
  
  /**
   * @brief 预分配内存块
   * @param count 要预分配的内存块数量
   */
  void Preallocate(size_t count);
  
  /**
   * @brief 清理所有空闲内存块
   */
  void Clear();
  
  /**
   * @brief 获取当前池大小
   */
  size_t GetPoolSize() const;
  
  /**
   * @brief 获取空闲内存块数量
   */
  size_t GetFreeCount() const;
  
  /**
   * @brief 获取已分配内存块数量
   */
  size_t GetAllocatedCount() const;

 private:
  /**
   * @brief 扩展内存池
   * @param count 要添加的内存块数量
   */
  void Expand(size_t count);

 private:
  size_t block_size_;           // 单个内存块大小
  float growth_factor_;         // 增长因子
  
  std::vector<T*> free_blocks_; // 空闲内存块
  std::vector<T*> all_blocks_;  // 所有内存块
  
  size_t allocated_count_;      // 已分配数量
  mutable std::mutex mutex_;    // 互斥锁
};

/**
 * @brief 内存池管理器
 * 
 * 管理多个不同大小的内存池
 */
class MemoryPoolManager {
 public:
  static MemoryPoolManager& GetInstance();
  
  /**
   * @brief 获取或创建内存池
   * @param block_size 内存块大小
   * @param block_count 初始块数量
   * @return 内存池指针
   */
  template <typename T>
  std::shared_ptr<MemoryPool<T>> GetPool(size_t block_size, size_t block_count = 10);
  
  /**
   * @brief 清理所有内存池
   */
  void ClearAll();
  
  /**
   * @brief 获取内存使用统计
   */
  struct MemoryStats {
    size_t total_memory;       // 总内存
    size_t used_memory;        // 已使用内存
    size_t free_memory;        // 空闲内存
    size_t pool_count;         // 内存池数量
  };
  MemoryStats GetMemoryStats();

 private:
  MemoryPoolManager() = default;
  ~MemoryPoolManager() = default;
  
  // 禁止拷贝
  MemoryPoolManager(const MemoryPoolManager&) = delete;
  MemoryPoolManager& operator=(const MemoryPoolManager&) = delete;

 private:
  // 内存池映射表
  std::unordered_map<size_t, std::shared_ptr<void>> pools_;
  mutable std::mutex mutex_;
};

/**
 * @brief 内存块包装器
 * 
 * 自动管理内存块的生命周期
 */
template <typename T>
class MemoryBlock {
 public:
  MemoryBlock(MemoryPool<T>* pool, T* data) 
      : pool_(pool), data_(data) {}
  
  ~MemoryBlock() {
    if (pool_ && data_) {
      pool_->Deallocate(data_);
    }
  }
  
  // 禁止拷贝
  MemoryBlock(const MemoryBlock&) = delete;
  MemoryBlock& operator=(const MemoryBlock&) = delete;
  
  // 允许移动
  MemoryBlock(MemoryBlock&& other) noexcept 
      : pool_(other.pool_), data_(other.data_) {
    other.pool_ = nullptr;
    other.data_ = nullptr;
  }
  
  MemoryBlock& operator=(MemoryBlock&& other) noexcept {
    if (this != &other) {
      if (pool_ && data_) {
        pool_->Deallocate(data_);
      }
      pool_ = other.pool_;
      data_ = other.data_;
      other.pool_ = nullptr;
      other.data_ = nullptr;
    }
    return *this;
  }
  
  // 访问操作符
  T& operator*() { return *data_; }
  const T& operator*() const { return *data_; }
  
  T* operator->() { return data_; }
  const T* operator->() const { return data_; }
  
  // 数据指针
  T* data() { return data_; }
  const T* data() const { return data_; }
  
  // 有效性检查
  bool valid() const { return data_ != nullptr; }

 private:
  MemoryPool<T>* pool_;
  T* data_;
};

// ==================== 模板实现 ====================

template <typename T>
MemoryPool<T>::MemoryPool(size_t block_size, size_t block_count, float growth_factor)
    : block_size_(block_size),
      growth_factor_(growth_factor),
      allocated_count_(0) {
  Preallocate(block_count);
}

template <typename T>
MemoryPool<T>::~MemoryPool() {
  Clear();
}

template <typename T>
T* MemoryPool<T>::Allocate() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (free_blocks_.empty()) {
    // 内存池已满，扩展
    size_t expand_count = static_cast<size_t>(all_blocks_.size() * growth_factor_);
    if (expand_count < 1) {
      expand_count = 1;
    }
    Expand(expand_count);
  }
  
  T* ptr = free_blocks_.back();
  free_blocks_.pop_back();
  allocated_count_++;
  
  return ptr;
}

template <typename T>
void MemoryPool<T>::Deallocate(T* ptr) {
  if (!ptr) {
    return;
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  // 检查指针是否属于此内存池
  auto it = std::find(all_blocks_.begin(), all_blocks_.end(), ptr);
  if (it == all_blocks_.end()) {
    throw std::invalid_argument("Pointer not from this memory pool");
  }
  
  free_blocks_.push_back(ptr);
  allocated_count_--;
}

template <typename T>
void MemoryPool<T>::Preallocate(size_t count) {
  std::lock_guard<std::mutex> lock(mutex_);
  Expand(count);
}

template <typename T>
void MemoryPool<T>::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  for (T* block : all_blocks_) {
    delete[] block;
  }
  
  free_blocks_.clear();
  all_blocks_.clear();
  allocated_count_ = 0;
}

template <typename T>
void MemoryPool<T>::Expand(size_t count) {
  for (size_t i = 0; i < count; ++i) {
    T* block = new T[block_size_];
    all_blocks_.push_back(block);
    free_blocks_.push_back(block);
  }
}

template <typename T>
size_t MemoryPool<T>::GetPoolSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return all_blocks_.size();
}

template <typename T>
size_t MemoryPool<T>::GetFreeCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return free_blocks_.size();
}

template <typename T>
size_t MemoryPool<T>::GetAllocatedCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocated_count_;
}

template <typename T>
std::shared_ptr<MemoryPool<T>> MemoryPoolManager::GetPool(size_t block_size, size_t block_count) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t key = block_size;
  auto it = pools_.find(key);
  
  if (it == pools_.end()) {
    auto pool = std::make_shared<MemoryPool<T>>(block_size, block_count);
    pools_[key] = pool;
    return pool;
  }
  
  return std::static_pointer_cast<MemoryPool<T>>(it->second);
}

}  // namespace distributed_inference
