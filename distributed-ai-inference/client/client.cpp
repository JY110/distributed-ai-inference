#include "../build/inference_service.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using namespace distributed_inference;
using namespace grpc;
using namespace cv;

class InferenceClient {
 public:
  InferenceClient(const std::string& server_address) {
    channel_ = CreateChannel(server_address, InsecureChannelCredentials());
    stub_ = distributed_inference::InferenceService::NewStub(channel_);
    spdlog::info("Connected to server: {}", server_address);
  }

  // 测试单张图片推理
  void TestSingleImageInference(const std::string& image_path, 
                               const std::string& model_name) {
    spdlog::info("Testing single image inference with model: {}", model_name);
    
    // 加载图片
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      spdlog::error("Failed to load image: {}", image_path);
      return;
    }
    
    // 构建请求
    distributed_inference::PredictRequest request;
    request.set_request_id("test-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
    request.set_model_name(model_name);
    request.set_model_version("latest");
    request.set_task_type(TASK_TYPE_DETECTION);
    request.set_input_type(INPUT_TYPE_IMAGE);
    
    // 编码图片
  std::vector<unsigned char> buffer;
  cv::imencode(".jpg", image, buffer);
    
    auto* image_input = request.mutable_image();
    image_input->set_image_data(buffer.data(), buffer.size());
    image_input->set_format(IMAGE_FORMAT_JPEG);
    
    // 发送请求
    distributed_inference::PredictResponse response;
    ClientContext context;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    Status status = stub_->Predict(&context, request, &response);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    int64_t total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    if (status.ok() && response.success()) {
      spdlog::info("Single inference successful");
      spdlog::info("Request ID: {}", response.request_id());
      spdlog::info("Inference time: {}ms", response.inference_time_ms());
      spdlog::info("Total time: {}ms", total_time_ms);
      
      // 解析检测结果
      if (response.has_detection()) {
        const auto& detection = response.detection();
        spdlog::info("Detected {} objects", detection.boxes_size());
        
        for (int i = 0; i < detection.boxes_size(); ++i) {
          const auto& box = detection.boxes(i);
          spdlog::info("Object {}: class={}, confidence={:.2f}, box=({:.2f}, {:.2f}, {:.2f}, {:.2f})",
                      i+1, box.class_name(), box.confidence(),
                      box.x(), box.y(), box.width(), box.height());
        }
      }
    } else {
      spdlog::error("Inference failed: {}", response.error_message());
      if (!status.ok()) {
        spdlog::error("gRPC error: {}", status.error_message());
      }
    }
  }

  // 测试批量推理
  void TestBatchInference(const std::vector<std::string>& image_paths, 
                         const std::string& model_name) {
    spdlog::info("Testing batch inference with model: {}, batch size: {}", 
                 model_name, image_paths.size());
    
    distributed_inference::BatchPredictRequest request;
    request.set_batch_id("batch-test-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
    
    for (const auto& image_path : image_paths) {
      cv::Mat image = cv::imread(image_path);
      if (image.empty()) {
        spdlog::error("Failed to load image: {}", image_path);
        continue;
      }
      
      auto* single_request = request.add_requests();
      single_request->set_request_id("batch-" + std::to_string(request.requests_size()));
      single_request->set_model_name(model_name);
      single_request->set_model_version("latest");
      single_request->set_task_type(TASK_TYPE_DETECTION);
      single_request->set_input_type(INPUT_TYPE_IMAGE);
      
      std::vector<unsigned char> buffer;
      cv::imencode(".jpg", image, buffer);
      
      auto* image_input = single_request->mutable_image();
      image_input->set_image_data(buffer.data(), buffer.size());
      image_input->set_format(IMAGE_FORMAT_JPEG);
    }
    
    distributed_inference::BatchPredictResponse response;
    ClientContext context;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    Status status = stub_->BatchPredict(&context, request, &response);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    int64_t total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    if (status.ok() && response.success()) {
      spdlog::info("Batch inference successful");
      spdlog::info("Batch ID: {}", response.batch_id());
      spdlog::info("Total time: {}ms", total_time_ms);
      spdlog::info("Total inference time: {}ms", response.total_inference_time_ms());
      spdlog::info("Success count: {}, Failed count: {}", 
                   response.success_count(), response.failed_count());
      
      for (int i = 0; i < response.responses_size(); ++i) {
        const auto& single_response = response.responses(i);
        if (single_response.success()) {
          spdlog::info("Request {}: successful, time: {}ms", 
                       i+1, single_response.inference_time_ms());
        } else {
          spdlog::error("Request {}: failed - {}", 
                       i+1, single_response.error_message());
        }
      }
    } else {
      spdlog::error("Batch inference failed: {}", response.error_message());
      if (!status.ok()) {
        spdlog::error("gRPC error: {}", status.error_message());
      }
    }
  }

  // 测试健康检查
  void TestHealthCheck() {
    spdlog::info("Testing health check");
    
    distributed_inference::HealthCheckRequest request;
    distributed_inference::HealthCheckResponse response;
    ClientContext context;
    
    Status status = stub_->HealthCheck(&context, request, &response);
    
    if (status.ok()) {
      spdlog::info("Health check successful");
      spdlog::info("Service ID: {}", response.service_id());
      spdlog::info("Status: {}", response.status());
      spdlog::info("Version: {}", response.version());
      spdlog::info("Timestamp: {}", response.timestamp());
    } else {
      spdlog::error("Health check failed: {}", status.error_message());
    }
  }

  // 测试服务状态
  void TestServiceStatus() {
    spdlog::info("Testing service status");
    
    distributed_inference::ServiceStatusRequest request;
    request.set_include_model_status(true);
    
    distributed_inference::ServiceStatusResponse response;
    ClientContext context;
    
    Status status = stub_->GetServiceStatus(&context, request, &response);
    
    if (status.ok()) {
      spdlog::info("Service status successful");
      spdlog::info("Service ID: {}", response.service_id());
      spdlog::info("Service version: {}", response.service_version());
      spdlog::info("Uptime: {} seconds", response.uptime_seconds());
      spdlog::info("Total requests: {}", response.total_requests());
      spdlog::info("Active requests: {}", response.active_requests());
      spdlog::info("CPU usage: {:.2f}%", response.cpu_usage_percent());
      spdlog::info("Memory usage: {:.2f}%", response.memory_usage_percent());
      
      spdlog::info("Model status:");
      for (int i = 0; i < response.model_status_size(); ++i) {
        const auto& model = response.model_status(i);
        spdlog::info("  Model: {} v{}, loaded: {}, inference count: {}", 
                     model.model_name(), model.model_version(), 
                     model.is_loaded() ? "yes" : "no", 
                     model.inference_count());
      }
    } else {
      spdlog::error("Service status failed: {}", status.error_message());
    }
  }

  // 测试模型加载
  void TestLoadModel(const std::string& model_name, 
                    const std::string& model_path) {
    spdlog::info("Testing load model: {}", model_name);
    
    distributed_inference::LoadModelRequest request;
    request.set_model_name(model_name);
    request.set_model_version("latest");
    request.set_model_path(model_path);
    
    distributed_inference::LoadModelResponse response;
    ClientContext context;
    
    Status status = stub_->LoadModel(&context, request, &response);
    
    if (status.ok() && response.success()) {
      spdlog::info("Load model successful");
      spdlog::info("Model: {} v{}", response.model_name(), response.model_version());
      spdlog::info("Load time: {}ms", response.load_time_ms());
    } else {
      spdlog::error("Load model failed: {}", response.error_message());
      if (!status.ok()) {
        spdlog::error("gRPC error: {}", status.error_message());
      }
    }
  }

 private:
  std::shared_ptr<Channel> channel_;
  std::unique_ptr<distributed_inference::InferenceService::Stub> stub_;
};

int main(int argc, char* argv[]) {
  // 配置spdlog
  spdlog::set_level(spdlog::level::info);
  
  if (argc < 2) {
    std::cerr << "Usage: client <server_address> [image_path] [model_name]" << std::endl;
    return 1;
  }
  
  std::string server_address = argv[1];
  std::string image_path = argc > 2 ? argv[2] : "test.jpg";
  std::string model_name = argc > 3 ? argv[3] : "yolov5s";
  
  try {
    InferenceClient client(server_address);
    
    // 测试健康检查
    client.TestHealthCheck();
    std::cout << "\n";
    
    // 测试服务状态
    client.TestServiceStatus();
    std::cout << "\n";
    
    // 测试模型加载
    client.TestLoadModel(model_name, "/home/ubuntu2404/c++/distributed-ai-inference/models/resnet50-v1-12.onnx");
    std::cout << "\n";
    
    // 测试单张图片推理
    client.TestSingleImageInference(image_path, model_name);
    std::cout << "\n";
    
    // 测试批量推理
    std::vector<std::string> batch_images = {image_path, image_path};
    client.TestBatchInference(batch_images, model_name);
    
  } catch (const std::exception& e) {
    spdlog::error("Client error: {}", e.what());
    return 1;
  }
  
  return 0;
}
