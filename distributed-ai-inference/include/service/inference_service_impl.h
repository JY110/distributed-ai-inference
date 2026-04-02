#pragma once

#include "../../build/inference_service.grpc.pb.h"
#include "../core/onnx_runtime_engine.h"
#include <grpc/grpc.h>
#include <grpcpp/server_context.h>
#include <grpcpp/server.h>
#include <memory>
#include <atomic>
#include <chrono>
#include <string>

namespace distributed_inference {

/**
 * @brief gRPC推理服务实现类
 * 
 * 设计思路：
 * 1. 实现.proto中定义的所有RPC方法
 * 2. 集成ONNX Runtime推理引擎
 * 3. 实现请求参数校验和错误处理
 * 4. 记录全链路日志和性能指标
 * 5. 支持并发请求处理
 */
class InferenceServiceImpl final : public distributed_inference::InferenceService::Service {
 public:
  explicit InferenceServiceImpl(const std::string& service_id);
  virtual ~InferenceServiceImpl() = default;

  /**
   * @brief 单张图片推理
   */
  grpc::Status Predict(grpc::ServerContext* context,
                       const distributed_inference::PredictRequest* request,
                       distributed_inference::PredictResponse* response) override;

  /**
   * @brief 批量图片推理
   */
  grpc::Status BatchPredict(grpc::ServerContext* context,
                            const distributed_inference::BatchPredictRequest* request,
                            distributed_inference::BatchPredictResponse* response) override;

  /**
   * @brief 健康检查
   */
  grpc::Status HealthCheck(grpc::ServerContext* context,
                           const distributed_inference::HealthCheckRequest* request,
                           distributed_inference::HealthCheckResponse* response) override;

  /**
   * @brief 获取服务状态
   */
  grpc::Status GetServiceStatus(grpc::ServerContext* context,
                                const distributed_inference::ServiceStatusRequest* request,
                                distributed_inference::ServiceStatusResponse* response) override;

  /**
   * @brief 加载模型
   */
  grpc::Status LoadModel(grpc::ServerContext* context,
                         const distributed_inference::LoadModelRequest* request,
                         distributed_inference::LoadModelResponse* response) override;

  /**
   * @brief 卸载模型
   */
  grpc::Status UnloadModel(grpc::ServerContext* context,
                           const distributed_inference::UnloadModelRequest* request,
                           distributed_inference::UnloadModelResponse* response) override;

  /**
   * @brief 获取服务ID
   */
  std::string GetServiceId() const { return service_id_; }

  /**
   * @brief 获取启动时间
   */
  std::chrono::steady_clock::time_point GetStartTime() const { return start_time_; }

  /**
   * @brief 获取总请求数
   */
  int64_t GetTotalRequests() const { return total_requests_.load(); }

  /**
   * @brief 获取活跃请求数
   */
  int32_t GetActiveRequests() const { return active_requests_.load(); }

 private:
  /**
   * @brief 验证预测请求
   */
  bool ValidatePredictRequest(const distributed_inference::PredictRequest* request, std::string* error_msg);

  /**
   * @brief 将OpenCV Mat转换为gRPC响应
   */
  void ConvertToGrpcResponse(const InferenceResult& infer_result,
                             distributed_inference::PredictResponse* grpc_response,
                             const distributed_inference::PredictRequest* request);

  /**
   * @brief 处理图片输入
   */
  InferenceResult ProcessImageInput(const distributed_inference::ImageInput& image_input,
                                     const std::string& model_name,
                                     const std::string& model_version);

  /**
   * @brief 处理张量输入
   */
  InferenceResult ProcessTensorInput(const distributed_inference::Tensor& tensor_input,
                                      const std::string& model_name,
                                      const std::string& model_version);

  /**
   * @brief 生成请求ID
   */
  std::string GenerateRequestId();

  /**
   * @brief 记录请求日志
   */
  void LogRequest(const std::string& method,
                  const std::string& request_id,
                  bool success,
                  int64_t latency_ms,
                  const std::string& error_msg = "");

 private:
  std::string service_id_;  // 服务唯一标识
  std::chrono::steady_clock::time_point start_time_;  // 启动时间
  
  // 统计信息
  std::atomic<int64_t> total_requests_{0};      // 总请求数
  std::atomic<int32_t> active_requests_{0};     // 活跃请求数
  std::atomic<int64_t> request_counter_{0};     // 请求计数器 (用于生成ID)
};

/**
 * @brief gRPC服务器类
 * 
 * 封装gRPC服务器的启动、停止和配置
 */
class GrpcServer {
 public:
  struct Config {
    std::string listen_address = "0.0.0.0:50051";  // 监听地址
    int max_workers = 4;                           // 工作线程数
    int max_concurrent_streams = 100;             // 最大并发流
    int keepalive_time_ms = 10000;                // keepalive时间
    int keepalive_timeout_ms = 5000;              // keepalive超时
  };

  explicit GrpcServer(const Config& config);
  ~GrpcServer();

  /**
   * @brief 启动服务器
   */
  bool Start();

  /**
   * @brief 停止服务器
   */
  void Stop();

  /**
   * @brief 等待服务器停止
   */
  void Wait();

  /**
   * @brief 检查服务器是否正在运行
   */
  bool IsRunning() const { return is_running_.load(); }

  /**
   * @brief 获取服务实现
   */
  std::shared_ptr<InferenceServiceImpl> GetService() { return service_impl_; }

 private:
  Config config_;
  std::atomic<bool> is_running_{false};
  std::shared_ptr<InferenceServiceImpl> service_impl_;
  std::unique_ptr<grpc::Server> server_;
};

}  // namespace distributed_inference