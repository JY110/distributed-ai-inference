#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <functional>

namespace distributed_inference {

/**
 * @brief 服务节点信息结构体
 */
struct ServiceNode {
  std::string node_id;              // 节点唯一标识
  std::string service_name;         // 服务名称
  std::string host;                 // 主机地址
  int port;                         // 端口号
  std::string version;              // 服务版本
  std::unordered_map<std::string, std::string> metadata;  // 元数据
  
  // 健康状态
  bool is_healthy = true;           // 是否健康
  int64_t last_heartbeat_time = 0;  // 最后一次心跳时间
  int consecutive_failures = 0;     // 连续失败次数
  
  // 性能指标
  float cpu_usage = 0.0f;           // CPU使用率
  float memory_usage = 0.0f;        // 内存使用率
  int active_requests = 0;          // 活跃请求数
  float avg_latency_ms = 0.0f;      // 平均延迟
  
  // 权重 (用于加权轮询负载均衡)
  int weight = 1;
  
  // 时间戳
  std::chrono::steady_clock::time_point register_time;
};

/**
 * @brief 服务注册与发现接口
 * 
 * 设计思路：
 * 1. 提供服务注册、注销、发现的基础接口
 * 2. 支持多种后端实现 (内存、Redis、etcd等)
 * 3. 支持服务健康状态监控
 */
class ServiceRegistry {
 public:
  virtual ~ServiceRegistry() = default;
  
  /**
   * @brief 注册服务节点
   */
  virtual bool Register(const ServiceNode& node) = 0;
  
  /**
   * @brief 注销服务节点
   */
  virtual bool Deregister(const std::string& node_id) = 0;
  
  /**
   * @brief 更新服务节点信息
   */
  virtual bool UpdateNode(const ServiceNode& node) = 0;
  
  /**
   * @brief 发送心跳
   */
  virtual bool Heartbeat(const std::string& node_id) = 0;
  
  /**
   * @brief 发现服务节点
   */
  virtual std::vector<ServiceNode> Discover(const std::string& service_name) = 0;
  
  /**
   * @brief 获取单个节点
   */
  virtual std::shared_ptr<ServiceNode> GetNode(const std::string& node_id) = 0;
  
  /**
   * @brief 订阅服务变化
   */
  virtual void Subscribe(const std::string& service_name,
                         std::function<void(const std::vector<ServiceNode>&)> callback) = 0;
  
  /**
   * @brief 标记节点不健康
   */
  virtual bool MarkNodeUnhealthy(const std::string& node_id) = 0;
};

/**
 * @brief 内存实现的服务注册中心
 * 
 * 适用于单节点部署或测试环境
 */
class InMemoryServiceRegistry : public ServiceRegistry {
 public:
  InMemoryServiceRegistry();
  ~InMemoryServiceRegistry() override = default;
  
  bool Register(const ServiceNode& node) override;
  bool Deregister(const std::string& node_id) override;
  bool UpdateNode(const ServiceNode& node) override;
  bool Heartbeat(const std::string& node_id) override;
  std::vector<ServiceNode> Discover(const std::string& service_name) override;
  std::shared_ptr<ServiceNode> GetNode(const std::string& node_id) override;
  void Subscribe(const std::string& service_name,
                 std::function<void(const std::vector<ServiceNode>&)> callback) override;
  
  /**
   * @brief 清理过期节点
   * @param timeout_seconds 超时时间(秒)
   */
  void CleanupExpiredNodes(int timeout_seconds);
  
  /**
   * @brief 标记节点不健康
   */
  bool MarkNodeUnhealthy(const std::string& node_id) override;
  
  /**
   * @brief 获取所有服务名称
   */
  std::vector<std::string> GetAllServiceNames();

 private:
  std::unordered_map<std::string, std::shared_ptr<ServiceNode>> nodes_;
  std::unordered_map<std::string, std::vector<std::string>> service_index_;
  mutable std::shared_mutex mutex_;
  
  // 订阅回调
  std::unordered_map<std::string, 
                     std::vector<std::function<void(const std::vector<ServiceNode>&)>>> callbacks_;
  mutable std::shared_mutex callback_mutex_;
};

/**
 * @brief 健康检查器
 * 
 * 定期检查服务节点健康状态
 */
class HealthChecker {
 public:
  struct Config {
    int check_interval_ms = 5000;      // 检查间隔
    int timeout_ms = 3000;             // 超时时间
    int max_failures = 3;              // 最大失败次数
    int recovery_interval_ms = 30000;  // 恢复检查间隔
  };
  
  explicit HealthChecker(const Config& config);
  ~HealthChecker();
  
  /**
   * @brief 启动健康检查
   */
  void Start(std::shared_ptr<ServiceRegistry> registry);
  
  /**
   * @brief 停止健康检查
   */
  void Stop();
  
  /**
   * @brief 添加检查目标
   */
  void AddTarget(const std::string& node_id, 
                 std::function<bool()> health_check_func);
  
  /**
   * @brief 移除检查目标
   */
  void RemoveTarget(const std::string& node_id);

 private:
  void CheckLoop();
  
 private:
  Config config_;
  std::shared_ptr<ServiceRegistry> registry_;
  std::unordered_map<std::string, std::function<bool()>> targets_;
  mutable std::mutex targets_mutex_;
  
  std::atomic<bool> is_running_{false};
  std::thread check_thread_;
};

/**
 * @brief 负载均衡器接口
 */
class LoadBalancer {
 public:
  virtual ~LoadBalancer() = default;
  
  /**
   * @brief 选择服务节点
   * @param service_name 服务名称
   * @param nodes 可用节点列表
   * @return 选中的节点，nullptr表示无可用节点
   */
  virtual std::shared_ptr<ServiceNode> Select(
      const std::string& service_name,
      const std::vector<ServiceNode>& nodes) = 0;
  
  /**
   * @brief 报告节点调用结果
   */
  virtual void ReportResult(const std::string& node_id, bool success, int64_t latency_ms) = 0;
};

/**
 * @brief 轮询负载均衡器
 */
class RoundRobinLoadBalancer : public LoadBalancer {
 public:
  std::shared_ptr<ServiceNode> Select(const std::string& service_name,
                                       const std::vector<ServiceNode>& nodes) override;
  void ReportResult(const std::string& node_id, bool success, int64_t latency_ms) override;

 private:
  std::unordered_map<std::string, size_t> counters_;
  mutable std::mutex mutex_;
};

/**
 * @brief 加权轮询负载均衡器
 */
class WeightedRoundRobinLoadBalancer : public LoadBalancer {
 public:
  std::shared_ptr<ServiceNode> Select(const std::string& service_name,
                                       const std::vector<ServiceNode>& nodes) override;
  void ReportResult(const std::string& node_id, bool success, int64_t latency_ms) override;

 private:
  struct NodeState {
    int current_weight = 0;
    int effective_weight;
  };
  
  std::unordered_map<std::string, NodeState> node_states_;
  mutable std::mutex mutex_;
};

/**
 * @brief 最少连接负载均衡器
 */
class LeastConnectionsLoadBalancer : public LoadBalancer {
 public:
  std::shared_ptr<ServiceNode> Select(const std::string& service_name,
                                       const std::vector<ServiceNode>& nodes) override;
  void ReportResult(const std::string& node_id, bool success, int64_t latency_ms) override;

 private:
  std::unordered_map<std::string, int> connection_counts_;
  mutable std::mutex mutex_;
};

/**
 * @brief 服务治理管理器
 * 
 * 整合服务注册、健康检查、负载均衡
 */
class ServiceGovernance {
 public:
  static ServiceGovernance& GetInstance();
  
  /**
   * @brief 初始化
   */
  bool Initialize(std::shared_ptr<ServiceRegistry> registry,
                  std::shared_ptr<LoadBalancer> load_balancer);
  
  /**
   * @brief 注册当前服务
   */
  bool RegisterSelf(const std::string& node_id,
                    const std::string& service_name,
                    const std::string& host,
                    int port,
                    const std::unordered_map<std::string, std::string>& metadata = {});
  
  /**
   * @brief 注销当前服务
   */
  bool DeregisterSelf();
  
  /**
   * @brief 发现服务
   */
  std::vector<ServiceNode> DiscoverService(const std::string& service_name);
  
  /**
   * @brief 选择服务节点
   */
  std::shared_ptr<ServiceNode> SelectNode(const std::string& service_name);
  
  /**
   * @brief 报告调用结果
   */
  void ReportCallResult(const std::string& node_id, bool success, int64_t latency_ms);
  
  /**
   * @brief 更新自身状态
   */
  void UpdateSelfStatus(float cpu_usage, float memory_usage, 
                        int active_requests, float avg_latency_ms);
  
  /**
   * @brief 获取当前节点ID
   */
  std::string GetSelfNodeId() const { return self_node_id_; }

 private:
  ServiceGovernance() = default;
  ~ServiceGovernance() = default;
  
 private:
  std::shared_ptr<ServiceRegistry> registry_;
  std::shared_ptr<LoadBalancer> load_balancer_;
  std::unique_ptr<HealthChecker> health_checker_;
  
  std::string self_node_id_;
  std::string self_service_name_;
  mutable std::mutex self_mutex_;
  
  std::atomic<bool> initialized_{false};
};

}  // namespace distributed_inference
