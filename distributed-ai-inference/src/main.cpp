#include "service/inference_service_impl.h"
#include "governance/service_registry.h"
#include "core/onnx_runtime_engine.h"
#include <spdlog/spdlog.h>
#include <signal.h>
#include <thread>
#include <chrono>

using namespace distributed_inference;

// 全局变量
std::unique_ptr<GrpcServer> g_server = nullptr;
std::atomic<bool> g_running = true;

// 信号处理函数
void SignalHandler(int signal) {
  spdlog::info("Received signal {}", signal);
  g_running = false;
  if (g_server) {
    g_server->Stop();
  }
}

// 监控线程
void MonitorThread() {
  while (g_running) {
    // 打印服务状态
    if (g_server && g_server->IsRunning()) {
      auto service = g_server->GetService();
      if (service) {
        spdlog::debug("Service status - total_requests: {}, active_requests: {}",
                     service->GetTotalRequests(), service->GetActiveRequests());
      }
    }
    
    // 每30秒检查一次
    std::this_thread::sleep_for(std::chrono::seconds(30));
  }
}

int main(int /*argc*/, char* /*argv*/[]) {
  // 配置日志
  spdlog::set_level(spdlog::level::info);
  
  spdlog::info("Starting distributed AI inference service...");
  
  // 注册信号处理
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);
  
  try {
    // 启动gRPC服务器
    GrpcServer::Config server_config;
    server_config.listen_address = "0.0.0.0:50051";
    server_config.max_workers = 8;
    server_config.max_concurrent_streams = 1000;
    
    g_server = std::make_unique<GrpcServer>(server_config);
    
    if (!g_server->Start()) {
      spdlog::error("Failed to start gRPC server");
      return 1;
    }
    
    // 启动监控线程
    std::thread monitor_thread(MonitorThread);
    
    spdlog::info("Inference service started successfully");
    spdlog::info("Listening on: {}", server_config.listen_address);
    
    // 等待服务器停止
    g_server->Wait();
    
    // 等待监控线程结束
    monitor_thread.join();
    
  } catch (const std::exception& e) {
    spdlog::error("Error: {}", e.what());
    return 1;
  }
  
  spdlog::info("Inference service stopped");
  return 0;
}