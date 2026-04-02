#include "../../include/core/onnx_runtime_engine.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <shared_mutex>

namespace distributed_inference {

// 静态成员初始化
std::shared_ptr<Ort::Env> OnnxRuntimeEngine::env_ = nullptr;
std::mutex OnnxRuntimeEngine::env_mutex_;
int OnnxRuntimeEngine::env_ref_count_ = 0;

OnnxRuntimeEngine::OnnxRuntimeEngine() {
  std::lock_guard<std::mutex> lock(env_mutex_);
  if (env_ref_count_ == 0) {
    // 首次创建，初始化ONNX Runtime环境
    OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
    env_ = std::make_shared<Ort::Env>(logging_level, "onnx_runtime");
    spdlog::info("ONNX Runtime environment initialized");
  }
  env_ref_count_++;
  allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
}

OnnxRuntimeEngine::~OnnxRuntimeEngine() {
  UnloadModel();
  {
    std::lock_guard<std::mutex> lock(env_mutex_);
    env_ref_count_--;
    if (env_ref_count_ == 0) {
      env_.reset();
      spdlog::info("ONNX Runtime environment released");
    }
  }
}

OnnxRuntimeEngine::OnnxRuntimeEngine(OnnxRuntimeEngine&& other) noexcept {
  session_ = std::move(other.session_);
  allocator_ = std::move(other.allocator_);
  config_ = std::move(other.config_);
  input_names_ = std::move(other.input_names_);
  output_names_ = std::move(other.output_names_);
  input_names_ptr_ = std::move(other.input_names_ptr_);
  output_names_ptr_ = std::move(other.output_names_ptr_);
  input_shape_ = std::move(other.input_shape_);
  output_shapes_ = std::move(other.output_shapes_);
  inference_count_ = other.inference_count_;
  total_inference_time_ms_ = other.total_inference_time_ms_;
  
  {
    std::lock_guard<std::mutex> lock(env_mutex_);
    env_ref_count_++;
  }
}

OnnxRuntimeEngine& OnnxRuntimeEngine::operator=(OnnxRuntimeEngine&& other) noexcept {
  if (this != &other) {
    UnloadModel();
    
    session_ = std::move(other.session_);
    allocator_ = std::move(other.allocator_);
    config_ = std::move(other.config_);
    input_names_ = std::move(other.input_names_);
    output_names_ = std::move(other.output_names_);
    input_names_ptr_ = std::move(other.input_names_ptr_);
    output_names_ptr_ = std::move(other.output_names_ptr_);
    input_shape_ = std::move(other.input_shape_);
    output_shapes_ = std::move(other.output_shapes_);
    inference_count_ = other.inference_count_;
    total_inference_time_ms_ = other.total_inference_time_ms_;
  }
  return *this;
}

bool OnnxRuntimeEngine::Initialize(const ModelConfig& config) {
  // 先设置配置，不获取锁
  config_ = config;
  return LoadModel(config.model_path, config);
}

Ort::SessionOptions OnnxRuntimeEngine::CreateSessionOptions(const ModelConfig& config) {
  Ort::SessionOptions session_options;
  
  // 设置线程数
  session_options.SetIntraOpNumThreads(config.intra_op_num_threads);
  session_options.SetInterOpNumThreads(config.inter_op_num_threads);
  
  // 图优化级别
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  
  // GPU支持 (如果启用)
  if (config.use_gpu) {
    // 尝试CUDA
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = config.gpu_device_id;
    try {
      session_options.AppendExecutionProvider_CUDA(cuda_options);
      spdlog::info("CUDA execution provider enabled, device_id: {}", config.gpu_device_id);
    } catch (const Ort::Exception& e) {
      spdlog::warn("Failed to enable CUDA: {}, falling back to CPU", e.what());
    }
  }
  
  return session_options;
}

bool OnnxRuntimeEngine::LoadModel(const std::string& model_path, const ModelConfig& config) {
  try {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = config;
    
    // 创建会话选项
    Ort::SessionOptions session_options = CreateSessionOptions(config);
    
    // 创建会话
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    // 获取输入信息
    size_t num_inputs = session_->GetInputCount();
    input_names_.clear();
    input_names_ptr_.clear();
    
    for (size_t i = 0; i < num_inputs; ++i) {
      Ort::AllocatedStringPtr name_ptr = session_->GetInputNameAllocated(i, *allocator_);
      std::string name(name_ptr.get());
      input_names_.push_back(name);
      input_names_ptr_.push_back(input_names_.back().c_str());
      
      // 获取输入形状
      Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      input_shape_ = tensor_info.GetShape();
      
      spdlog::info("Input {}: name={}, shape=[{}]", i, name, 
                   fmt::join(input_shape_, ", "));
    }
    
    // 获取输出信息
    size_t num_outputs = session_->GetOutputCount();
    output_names_.clear();
    output_names_ptr_.clear();
    output_shapes_.clear();
    
    for (size_t i = 0; i < num_outputs; ++i) {
      Ort::AllocatedStringPtr name_ptr = session_->GetOutputNameAllocated(i, *allocator_);
      std::string name(name_ptr.get());
      output_names_.push_back(name);
      output_names_ptr_.push_back(output_names_.back().c_str());
      
      // 获取输出形状
      Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      output_shapes_.push_back(tensor_info.GetShape());
      
      spdlog::info("Output {}: name={}, shape=[{}]", i, name,
                   fmt::join(output_shapes_.back(), ", "));
    }
    
    spdlog::info("Model loaded successfully: {}", model_path);
    return true;
    
  } catch (const Ort::Exception& e) {
    spdlog::error("Failed to load model {}: {}", model_path, e.what());
    return false;
  } catch (const std::exception& e) {
    spdlog::error("Unexpected error loading model {}: {}", model_path, e.what());
    return false;
  }
}

void OnnxRuntimeEngine::UnloadModel() {
  std::lock_guard<std::mutex> lock(config_mutex_);
  if (session_) {
    session_.reset();
    input_names_.clear();
    output_names_.clear();
    input_names_ptr_.clear();
    output_names_ptr_.clear();
    input_shape_.clear();
    output_shapes_.clear();
    spdlog::info("Model unloaded: {}", config_.model_name);
  }
}

bool OnnxRuntimeEngine::IsModelLoaded() const {
  return session_ != nullptr;
}

std::vector<float> OnnxRuntimeEngine::Preprocess(const cv::Mat& image) {
  std::lock_guard<std::mutex> lock(config_mutex_);
  
  // 获取目标尺寸
  int target_w = config_.target_width;
  int target_h = config_.target_height;
  
  // 调整图片大小
  cv::Mat resized;
  if (config_.keep_aspect_ratio) {
    // 保持宽高比，填充黑边
    float scale_x = static_cast<float>(target_w) / image.cols;
    float scale_y = static_cast<float>(target_h) / image.rows;
    float scale = std::min(scale_x, scale_y);
    
    int new_w = static_cast<int>(image.cols * scale);
    int new_h = static_cast<int>(image.rows * scale);
    
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(new_w, new_h));
    
    // 创建目标大小的画布并填充
    resized = cv::Mat(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Rect roi((target_w - new_w) / 2, (target_h - new_h) / 2, new_w, new_h);
    scaled.copyTo(resized(roi));
  } else {
    // 直接拉伸
    cv::resize(image, resized, cv::Size(target_w, target_h));
  }
  
  // 转换为float32并归一化
  cv::Mat float_mat;
  resized.convertTo(float_mat, CV_32FC3, config_.scale);
  
  // 减均值除标准差
  std::vector<cv::Mat> channels(3);
  cv::split(float_mat, channels);
  
  for (int c = 0; c < 3; ++c) {
    channels[c] = (channels[c] - config_.mean[c]) / config_.std[c];
  }
  
  // 合并通道
  cv::Mat normalized;
  cv::merge(channels, normalized);
  
  // HWC -> CHW 格式转换
  std::vector<float> input_tensor;
  input_tensor.reserve(3 * target_h * target_w);
  
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < target_h; ++h) {
      for (int w = 0; w < target_w; ++w) {
        input_tensor.push_back(normalized.at<cv::Vec3f>(h, w)[c]);
      }
    }
  }
  
  return input_tensor;
}

InferenceResult OnnxRuntimeEngine::Infer(const cv::Mat& image) {
  InferenceResult result;
  
  if (!IsModelLoaded()) {
    result.success = false;
    result.error_message = "Model not loaded";
    return result;
  }
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  try {
    // 预处理
    std::vector<float> input_data = Preprocess(image);
    
    // 构建输入张量形状 (batch_size=1)
    std::vector<int64_t> input_shape = input_shape_;
    if (!input_shape.empty()) {
      input_shape[0] = 1;  // batch size
    }
    
    // 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), 
        input_shape.size());
    
    // 执行推理
    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_ptr_.data(), &input_tensor, input_names_ptr_.size(),
        output_names_ptr_.data(), output_names_ptr_.size());
    
    // 计算推理时间
    auto end_time = std::chrono::high_resolution_clock::now();
    result.inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // 后处理 (根据任务类型)
    // 这里简化处理，实际应根据模型输出格式进行相应后处理
    result.success = true;
    
    // 更新统计
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      inference_count_++;
      total_inference_time_ms_ += result.inference_time_ms;
    }
    
  } catch (const Ort::Exception& e) {
    result.success = false;
    result.error_message = std::string("ONNX Runtime error: ") + e.what();
    spdlog::error("Inference failed: {}", e.what());
  } catch (const std::exception& e) {
    result.success = false;
    result.error_message = std::string("Error: ") + e.what();
    spdlog::error("Inference failed: {}", e.what());
  }
  
  return result;
}

InferenceResult OnnxRuntimeEngine::Infer(std::vector<float>& input_data,
                                          const std::vector<int64_t>& shape) {
  InferenceResult result;
  
  if (!IsModelLoaded()) {
    result.success = false;
    result.error_message = "Model not loaded";
    return result;
  }
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  try {
    // 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), shape.data(), 
        shape.size());
    
    // 执行推理
    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_ptr_.data(), &input_tensor, input_names_ptr_.size(),
        output_names_ptr_.data(), output_names_ptr_.size());
    
    // 计算推理时间
    auto end_time = std::chrono::high_resolution_clock::now();
    result.inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    result.success = true;
    
    // 更新统计
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      inference_count_++;
      total_inference_time_ms_ += result.inference_time_ms;
    }
    
  } catch (const Ort::Exception& e) {
    result.success = false;
    result.error_message = std::string("ONNX Runtime error: ") + e.what();
    spdlog::error("Inference failed: {}", e.what());
  }
  
  return result;
}

std::vector<InferenceResult> OnnxRuntimeEngine::BatchInfer(
    const std::vector<cv::Mat>& images) {
  std::vector<InferenceResult> results;
  results.reserve(images.size());
  
  // 简化实现：串行处理
  // 实际生产环境应实现真正的批量推理
  for (const auto& image : images) {
    results.push_back(Infer(image));
  }
  
  return results;
}

std::string OnnxRuntimeEngine::GetModelInfo() const {
  std::lock_guard<std::mutex> lock(config_mutex_);
  std::string info = "Model: " + config_.model_name + 
                     " v" + config_.model_version +
                     "\nPath: " + config_.model_path;
  return info;
}

std::vector<int64_t> OnnxRuntimeEngine::GetInputShape() const {
  std::lock_guard<std::mutex> lock(config_mutex_);
  return input_shape_;
}

std::vector<std::vector<int64_t>> OnnxRuntimeEngine::GetOutputShapes() const {
  std::lock_guard<std::mutex> lock(config_mutex_);
  return output_shapes_;
}

// ==================== ModelManager 实现 ====================

// 静态成员初始化（如果有的话）
// 注意：ModelManager类的成员变量是实例变量，不是静态变量，所以不需要在这里初始化

ModelManager& ModelManager::GetInstance() {
  static ModelManager instance;
  return instance;
}

bool ModelManager::RegisterModel(const std::string& model_name,
                                  const std::string& model_version,
                                  const ModelConfig& config) {
  std::unique_lock<std::shared_mutex> lock(configs_mutex_);
  configs_[model_name][model_version] = config;
  spdlog::info("Model registered: {} v{}", model_name, model_version);
  return true;
}

bool ModelManager::LoadModel(const std::string& model_name,
                              const std::string& model_version) {
  ModelConfig config;
  {
    std::shared_lock<std::shared_mutex> lock(configs_mutex_);
    auto model_it = configs_.find(model_name);
    if (model_it == configs_.end()) {
      spdlog::error("Model config not found: {}", model_name);
      return false;
    }
    auto version_it = model_it->second.find(model_version);
    if (version_it == model_it->second.end()) {
      spdlog::error("Model version not found: {} v{}", model_name, model_version);
      return false;
    }
    config = version_it->second;
  }
  
  auto engine = std::make_shared<OnnxRuntimeEngine>();
  if (!engine->Initialize(config)) {
    spdlog::error("Failed to initialize engine for {} v{}", model_name, model_version);
    return false;
  }
  
  {
    std::unique_lock<std::shared_mutex> lock(engines_mutex_);
    engines_[model_name][model_version] = engine;
  }
  
  spdlog::info("Model loaded: {} v{}", model_name, model_version);
  return true;
}

bool ModelManager::UnloadModel(const std::string& model_name,
                                const std::string& model_version) {
  std::unique_lock<std::shared_mutex> lock(engines_mutex_);
  auto model_it = engines_.find(model_name);
  if (model_it == engines_.end()) {
    return false;
  }
  
  auto version_it = model_it->second.find(model_version);
  if (version_it == model_it->second.end()) {
    return false;
  }
  
  version_it->second->UnloadModel();
  model_it->second.erase(version_it);
  
  if (model_it->second.empty()) {
    engines_.erase(model_it);
  }
  
  spdlog::info("Model unloaded: {} v{}", model_name, model_version);
  return true;
}

std::shared_ptr<OnnxRuntimeEngine> ModelManager::GetEngine(
    const std::string& model_name,
    const std::string& model_version) {
  std::shared_lock<std::shared_mutex> lock(engines_mutex_);
  auto model_it = engines_.find(model_name);
  if (model_it == engines_.end()) {
    return nullptr;
  }
  auto version_it = model_it->second.find(model_version);
  if (version_it == model_it->second.end()) {
    return nullptr;
  }
  return version_it->second;
}

std::shared_ptr<OnnxRuntimeEngine> ModelManager::GetDefaultEngine() {
  std::string name, version;
  {
    std::shared_lock<std::shared_mutex> lock(default_mutex_);
    name = default_model_name_;
    version = default_model_version_;
  }
  if (name.empty()) {
    return nullptr;
  }
  return GetEngine(name, version);
}

std::vector<std::pair<std::string, std::string>> ModelManager::ListLoadedModels() {
  std::shared_lock<std::shared_mutex> lock(engines_mutex_);
  std::vector<std::pair<std::string, std::string>> result;
  for (const auto& [model_name, versions] : engines_) {
    for (const auto& [version, engine] : versions) {
      if (engine && engine->IsModelLoaded()) {
        result.emplace_back(model_name, version);
      }
    }
  }
  return result;
}

std::vector<ModelManager::ModelStatus> ModelManager::GetAllModelStatus() {
  std::shared_lock<std::shared_mutex> lock(engines_mutex_);
  std::vector<ModelStatus> result;
  
  for (const auto& [model_name, versions] : engines_) {
    for (const auto& [version, engine] : versions) {
      ModelStatus status;
      status.model_name = model_name;
      status.model_version = version;
      status.is_loaded = engine && engine->IsModelLoaded();
      // 这里可以扩展更多统计信息
      result.push_back(status);
    }
  }
  
  return result;
}

}  // namespace distributed_inference
