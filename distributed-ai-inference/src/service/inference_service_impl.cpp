#include "../../include/service/inference_service_impl.h"
#include <spdlog/spdlog.h>
#include <opencv2/imgcodecs.hpp>
#include <grpcpp/server_builder.h>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace distributed_inference {

InferenceServiceImpl::InferenceServiceImpl(const std::string& service_id)
    : service_id_(service_id), start_time_(std::chrono::steady_clock::now()) {
  spdlog::info("InferenceServiceImpl created, service_id: {}", service_id_);
}

grpc::Status InferenceServiceImpl::Predict(grpc::ServerContext* /*context*/,
                                            const distributed_inference::PredictRequest* request,
                                            distributed_inference::PredictResponse* response) {
  auto request_start = std::chrono::high_resolution_clock::now();
  active_requests_++;
  total_requests_++;
  
  // 生成或复用请求ID
  std::string request_id = request->request_id();
  if (request_id.empty()) {
    request_id = GenerateRequestId();
  }
  response->set_request_id(request_id);
  
  spdlog::info("Predict request received, request_id: {}, model: {}:{}",
               request_id, request->model_name(), request->model_version());
  
  // 参数校验
  std::string error_msg;
  if (!ValidatePredictRequest(request, &error_msg)) {
    response->set_success(false);
    response->set_error_message(error_msg);
    active_requests_--;
    LogRequest("Predict", request_id, false, 0, error_msg);
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, error_msg);
  }
  
  // 执行推理
  InferenceResult infer_result;
  
  try {
    switch (request->input_type()) {
      case INPUT_TYPE_IMAGE:
        if (!request->has_image()) {
          throw std::invalid_argument("Input type is IMAGE but no image data provided");
        }
        infer_result = ProcessImageInput(request->image(), 
                                          request->model_name(),
                                          request->model_version());
        break;
        
      case INPUT_TYPE_TENSOR:
        if (!request->has_tensor()) {
          throw std::invalid_argument("Input type is TENSOR but no tensor data provided");
        }
        infer_result = ProcessTensorInput(request->tensor(),
                                           request->model_name(),
                                           request->model_version());
        break;
        
      default:
        throw std::invalid_argument("Unsupported input type");
    }
    
    // 转换为gRPC响应
    ConvertToGrpcResponse(infer_result, response, request);
    
  } catch (const std::exception& e) {
    spdlog::error("Predict failed, request_id: {}, error: {}", request_id, e.what());
    response->set_success(false);
    response->set_error_message(e.what());
    active_requests_--;
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - request_start).count();
    LogRequest("Predict", request_id, false, latency, e.what());
    return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
  }
  
  // 计算延迟
  auto request_end = std::chrono::high_resolution_clock::now();
  int64_t latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      request_end - request_start).count();
  
  response->set_inference_time_ms(infer_result.inference_time_ms);
  response->set_model_name(request->model_name());
  response->set_model_version(request->model_version());
  
  active_requests_--;
  LogRequest("Predict", request_id, true, latency_ms);
  
  return grpc::Status::OK;
}

grpc::Status InferenceServiceImpl::BatchPredict(grpc::ServerContext* /*context*/,
                                                 const distributed_inference::BatchPredictRequest* request,
                                                 distributed_inference::BatchPredictResponse* response) {
  auto batch_start = std::chrono::high_resolution_clock::now();
  
  response->set_batch_id(request->batch_id());
  
  spdlog::info("BatchPredict request received, batch_id: {}, count: {}",
               request->batch_id(), request->requests_size());
  
  int success_count = 0;
  int failed_count = 0;
  
  for (const auto& single_request : request->requests()) {
    PredictResponse* single_response = response->add_responses();
    
    grpc::ServerContext dummy_context;
    grpc::Status status = Predict(&dummy_context, &single_request, single_response);
    
    if (single_response->success()) {
      success_count++;
    } else {
      failed_count++;
    }
  }
  
  auto batch_end = std::chrono::high_resolution_clock::now();
  int64_t total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      batch_end - batch_start).count();
  
  response->set_success_count(success_count);
  response->set_failed_count(failed_count);
  response->set_total_inference_time_ms(total_time_ms);
  response->set_success(failed_count == 0 || request->allow_partial_success());
  
  spdlog::info("BatchPredict completed, batch_id: {}, success: {}, failed: {}, total_time: {}ms",
               request->batch_id(), success_count, failed_count, total_time_ms);
  
  return grpc::Status::OK;
}

grpc::Status InferenceServiceImpl::HealthCheck(grpc::ServerContext* /*context*/,
                                                const distributed_inference::HealthCheckRequest* /*request*/,
                                                distributed_inference::HealthCheckResponse* response) {
  response->set_service_id(service_id_);
  response->set_timestamp(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch()).count());
  response->set_version("1.0.0");
  
  // 检查模型是否已加载
  auto engine = ModelManager::GetInstance().GetDefaultEngine();
  if (engine && engine->IsModelLoaded()) {
    response->set_status(HealthCheckResponse::SERVING);
  } else {
    // 检查是否有任何已加载的模型
    auto loaded_models = ModelManager::GetInstance().ListLoadedModels();
    if (!loaded_models.empty()) {
      response->set_status(HealthCheckResponse::SERVING);
    } else {
      response->set_status(HealthCheckResponse::NOT_SERVING);
    }
  }
  
  return grpc::Status::OK;
}

grpc::Status InferenceServiceImpl::GetServiceStatus(grpc::ServerContext* /*context*/,
                                                     const distributed_inference::ServiceStatusRequest* request,
                                                     distributed_inference::ServiceStatusResponse* response) {
  response->set_service_id(service_id_);
  response->set_service_version("1.0.0");
  
  // 计算运行时间
  auto now = std::chrono::steady_clock::now();
  auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
  response->set_uptime_seconds(uptime.count());
  
  // 请求统计
  response->set_total_requests(static_cast<int32_t>(total_requests_.load()));
  response->set_active_requests(active_requests_.load());
  
  // CPU和内存使用率 (简化实现，实际应调用系统API)
  response->set_cpu_usage_percent(0.0f);
  response->set_memory_usage_percent(0.0f);
  
  // 模型状态
  if (request->include_model_status()) {
    ModelManager& model_manager = ModelManager::GetInstance();
    auto loaded_models = model_manager.ListLoadedModels();
    
    for (const auto& [model_name, version] : loaded_models) {
      auto* model_status = response->add_model_status();
      model_status->set_model_name(model_name);
      model_status->set_model_version(version);
      model_status->set_is_loaded(true);
      
      auto engine = model_manager.GetEngine(model_name, version);
      if (engine) {
        // 这里可以添加更多统计信息
        model_status->set_inference_count(0);
        model_status->set_avg_inference_time_ms(0.0f);
      }
    }
  }
  
  return grpc::Status::OK;
}

grpc::Status InferenceServiceImpl::LoadModel(grpc::ServerContext* /*context*/,
                                              const distributed_inference::LoadModelRequest* request,
                                              distributed_inference::LoadModelResponse* response) {
  spdlog::info("LoadModel request received, model: {}:{}, path: {}",
               request->model_name(), request->model_version(), request->model_path());
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // 构建模型配置
  ModelConfig config;
  config.model_name = request->model_name();
  config.model_version = request->model_version();
  config.model_path = request->model_path();
  
  // 解析额外配置
  for (const auto& [key, value] : request->config()) {
    if (key == "target_width") {
      config.target_width = std::stoi(value);
    } else if (key == "target_height") {
      config.target_height = std::stoi(value);
    } else if (key == "use_gpu") {
      config.use_gpu = (value == "true" || value == "1");
    } else if (key == "gpu_device_id") {
      config.gpu_device_id = std::stoi(value);
    } else if (key == "intra_op_num_threads") {
      config.intra_op_num_threads = std::stoi(value);
    } else if (key == "inter_op_num_threads") {
      config.inter_op_num_threads = std::stoi(value);
    }
  }
  
  // 注册并加载模型
  ModelManager::GetInstance().RegisterModel(
      request->model_name(), request->model_version(), config);
  
  bool success = ModelManager::GetInstance().LoadModel(
      request->model_name(), request->model_version());
  
  auto end_time = std::chrono::high_resolution_clock::now();
  int64_t load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();
  
  response->set_success(success);
  response->set_model_name(request->model_name());
  response->set_model_version(request->model_version());
  response->set_load_time_ms(load_time_ms);
  
  if (!success) {
    response->set_error_message("Failed to load model");
    spdlog::error("Failed to load model {}:{}", request->model_name(), request->model_version());
    return grpc::Status(grpc::StatusCode::INTERNAL, "Failed to load model");
  }
  
  spdlog::info("Model loaded successfully, model: {}:{}, time: {}ms",
               request->model_name(), request->model_version(), load_time_ms);
  
  return grpc::Status::OK;
}

grpc::Status InferenceServiceImpl::UnloadModel(grpc::ServerContext* /*context*/,
                                                const distributed_inference::UnloadModelRequest* request,
                                                distributed_inference::UnloadModelResponse* response) {
  spdlog::info("UnloadModel request received, model: {}:{}",
               request->model_name(), request->model_version());
  
  bool success = ModelManager::GetInstance().UnloadModel(
      request->model_name(), request->model_version());
  
  response->set_success(success);
  
  if (!success) {
    response->set_error_message("Model not found or already unloaded");
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "Model not found");
  }
  
  spdlog::info("Model unloaded successfully, model: {}:{}",
               request->model_name(), request->model_version());
  
  return grpc::Status::OK;
}

bool InferenceServiceImpl::ValidatePredictRequest(const PredictRequest* request,
                                                   std::string* error_msg) {
  if (request->model_name().empty()) {
    *error_msg = "Model name is required";
    return false;
  }
  
  if (request->input_type() == INPUT_TYPE_UNSPECIFIED) {
    *error_msg = "Input type must be specified";
    return false;
  }
  
  if (request->timeout_ms() < 0) {
    *error_msg = "Timeout must be non-negative";
    return false;
  }
  
  return true;
}

void InferenceServiceImpl::ConvertToGrpcResponse(const InferenceResult& infer_result,
                                                  PredictResponse* grpc_response,
                                                  const PredictRequest* request) {
  grpc_response->set_success(infer_result.success);
  
  if (!infer_result.success) {
    grpc_response->set_error_message(infer_result.error_message);
    return;
  }
  
  grpc_response->set_inference_time_ms(infer_result.inference_time_ms);
  
  // 根据任务类型填充不同的结果
  switch (request->task_type()) {
    case TASK_TYPE_DETECTION: {
      auto* detection = grpc_response->mutable_detection();
      for (const auto& det : infer_result.detections) {
        auto* box = detection->add_boxes();
        box->set_x(det.x);
        box->set_y(det.y);
        box->set_width(det.width);
        box->set_height(det.height);
        box->set_confidence(det.confidence);
        box->set_class_id(det.class_id);
        box->set_class_name(det.class_name);
      }
      break;
    }
    
    case TASK_TYPE_CLASSIFICATION: {
      if (!infer_result.classifications.empty()) {
        const auto& cls = infer_result.classifications[0];
        auto* classification = grpc_response->mutable_classification();
        classification->set_class_id(cls.class_id);
        classification->set_class_name(cls.class_name);
        classification->set_confidence(cls.confidence);
      }
      break;
    }
    
    default:
      // 返回原始张量输出
      // 这里简化处理
      break;
  }
}

InferenceResult InferenceServiceImpl::ProcessImageInput(const ImageInput& image_input,
                                                         const std::string& model_name,
                                                         const std::string& model_version) {
  // 解码图片
  std::vector<uchar> buffer(image_input.image_data().begin(), 
                            image_input.image_data().end());
  cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
  
  if (image.empty()) {
    throw std::runtime_error("Failed to decode image");
  }
  
  // 获取模型引擎
  std::shared_ptr<OnnxRuntimeEngine> engine;
  if (!model_name.empty()) {
    std::string version = model_version.empty() ? "latest" : model_version;
    engine = ModelManager::GetInstance().GetEngine(model_name, version);
  } else {
    engine = ModelManager::GetInstance().GetDefaultEngine();
  }
  
  if (!engine || !engine->IsModelLoaded()) {
    throw std::runtime_error("Model not loaded: " + model_name);
  }
  
  // 执行推理
  return engine->Infer(image);
}

InferenceResult InferenceServiceImpl::ProcessTensorInput(const Tensor& tensor_input,
                                                          const std::string& model_name,
                                                          const std::string& model_version) {
  // 转换张量数据
  std::vector<float> input_data(tensor_input.float_data().begin(),
                                 tensor_input.float_data().end());
  std::vector<int64_t> shape(tensor_input.shape().begin(),
                              tensor_input.shape().end());
  
  // 获取模型引擎
  std::shared_ptr<OnnxRuntimeEngine> engine;
  if (!model_name.empty()) {
    std::string version = model_version.empty() ? "latest" : model_version;
    engine = ModelManager::GetInstance().GetEngine(model_name, version);
  } else {
    engine = ModelManager::GetInstance().GetDefaultEngine();
  }
  
  if (!engine || !engine->IsModelLoaded()) {
    throw std::runtime_error("Model not loaded: " + model_name);
  }
  
  // 执行推理
  return engine->Infer(input_data, shape);
}

std::string InferenceServiceImpl::GenerateRequestId() {
  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
      now.time_since_epoch()).count();
  
  int64_t counter = request_counter_.fetch_add(1);
  
  std::stringstream ss;
  ss << service_id_ << "-" << timestamp << "-" << counter;
  return ss.str();
}

void InferenceServiceImpl::LogRequest(const std::string& method,
                                       const std::string& request_id,
                                       bool success,
                                       int64_t latency_ms,
                                       const std::string& error_msg) {
  if (success) {
    spdlog::info("Request completed, method: {}, request_id: {}, latency: {}ms",
                 method, request_id, latency_ms);
  } else {
    spdlog::error("Request failed, method: {}, request_id: {}, error: {}",
                  method, request_id, error_msg);
  }
}

// ==================== GrpcServer 实现 ====================

GrpcServer::GrpcServer(const Config& config) : config_(config) {
  service_impl_ = std::make_shared<InferenceServiceImpl>("inference-service-" + 
      std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
}

GrpcServer::~GrpcServer() {
  Stop();
}

bool GrpcServer::Start() {
  if (is_running_.load()) {
    spdlog::warn("Server is already running");
    return true;
  }
  
  grpc::ServerBuilder builder;
  
  // 监听地址
  builder.AddListeningPort(config_.listen_address, grpc::InsecureServerCredentials());
  
  // 注册服务
  builder.RegisterService(service_impl_.get());
  
  // 配置线程池
  builder.SetSyncServerOption(grpc::ServerBuilder::NUM_CQS, config_.max_workers);
  builder.SetSyncServerOption(grpc::ServerBuilder::MAX_POLLERS, config_.max_workers);
  
  // 构建并启动服务器
  server_ = builder.BuildAndStart();
  
  if (!server_) {
    spdlog::error("Failed to start gRPC server on {}", config_.listen_address);
    return false;
  }
  
  is_running_.store(true);
  spdlog::info("gRPC server started on {}", config_.listen_address);
  
  return true;
}

void GrpcServer::Stop() {
  if (!is_running_.load()) {
    return;
  }
  
  spdlog::info("Stopping gRPC server...");
  
  if (server_) {
    server_->Shutdown();
  }
  
  is_running_.store(false);
  spdlog::info("gRPC server stopped");
}

void GrpcServer::Wait() {
  if (server_) {
    server_->Wait();
  }
}

}  // namespace distributed_inference
