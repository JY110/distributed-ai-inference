#include "governance/service_registry.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace distributed_inference {

InMemoryServiceRegistry::InMemoryServiceRegistry() {
  spdlog::info("InMemoryServiceRegistry created");
}

bool InMemoryServiceRegistry::Register(const ServiceNode& node) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  
  auto it = nodes_.find(node.node_id);
  if (it != nodes_.end()) {
    spdlog::warn("Node already registered: {}", node.node_id);
    return false;
  }
  
  auto new_node = std::make_shared<ServiceNode>(node);
  new_node->register_time = std::chrono::steady_clock::now();
  new_node->last_heartbeat_time = 
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
  
  nodes_[node.node_id] = new_node;
  service_index_[node.service_name].push_back(node.node_id);
  
  spdlog::info("Node registered: {}, service: {}, {}:{}",
               node.node_id, node.service_name, node.host, node.port);
  
  // 通知订阅者
  { 
    std::shared_lock<std::shared_mutex> callback_lock(callback_mutex_);
    auto callback_it = callbacks_.find(node.service_name);
    if (callback_it != callbacks_.end()) {
      auto nodes = Discover(node.service_name);
      for (const auto& callback : callback_it->second) {
        callback(nodes);
      }
    }
  }
  
  return true;
}

bool InMemoryServiceRegistry::Deregister(const std::string& node_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  
  auto it = nodes_.find(node_id);
  if (it == nodes_.end()) {
    spdlog::warn("Node not found: {}", node_id);
    return false;
  }
  
  std::string service_name = it->second->service_name;
  nodes_.erase(it);
  
  // 从服务索引中移除
  auto service_it = service_index_.find(service_name);
  if (service_it != service_index_.end()) {
    auto& node_ids = service_it->second;
    auto node_it = std::find(node_ids.begin(), node_ids.end(), node_id);
    if (node_it != node_ids.end()) {
      node_ids.erase(node_it);
    }
    if (node_ids.empty()) {
      service_index_.erase(service_it);
    }
  }
  
  spdlog::info("Node deregistered: {}", node_id);
  
  // 通知订阅者
  { 
    std::shared_lock<std::shared_mutex> callback_lock(callback_mutex_);
    auto callback_it = callbacks_.find(service_name);
    if (callback_it != callbacks_.end()) {
      auto nodes = Discover(service_name);
      for (const auto& callback : callback_it->second) {
        callback(nodes);
      }
    }
  }
  
  return true;
}

bool InMemoryServiceRegistry::UpdateNode(const ServiceNode& node) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  
  auto it = nodes_.find(node.node_id);
  if (it == nodes_.end()) {
    spdlog::warn("Node not found: {}", node.node_id);
    return false;
  }
  
  // 更新节点信息
  it->second->service_name = node.service_name;
  it->second->host = node.host;
  it->second->port = node.port;
  it->second->version = node.version;
  it->second->metadata = node.metadata;
  it->second->is_healthy = node.is_healthy;
  it->second->cpu_usage = node.cpu_usage;
  it->second->memory_usage = node.memory_usage;
  it->second->active_requests = node.active_requests;
  it->second->avg_latency_ms = node.avg_latency_ms;
  it->second->weight = node.weight;
  
  spdlog::debug("Node updated: {}", node.node_id);
  
  return true;
}

bool InMemoryServiceRegistry::Heartbeat(const std::string& node_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  
  auto it = nodes_.find(node_id);
  if (it == nodes_.end()) {
    spdlog::warn("Heartbeat from unknown node: {}", node_id);
    return false;
  }
  
  it->second->last_heartbeat_time = 
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
  it->second->is_healthy = true;
  it->second->consecutive_failures = 0;
  
  spdlog::debug("Heartbeat received from: {}", node_id);
  
  return true;
}

std::vector<ServiceNode> InMemoryServiceRegistry::Discover(const std::string& service_name) {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  
  std::vector<ServiceNode> result;
  auto service_it = service_index_.find(service_name);
  if (service_it == service_index_.end()) {
    return result;
  }
  
  for (const auto& node_id : service_it->second) {
    auto node_it = nodes_.find(node_id);
    if (node_it != nodes_.end() && node_it->second->is_healthy) {
      result.push_back(*(node_it->second));
    }
  }
  
  return result;
}

std::shared_ptr<ServiceNode> InMemoryServiceRegistry::GetNode(const std::string& node_id) {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  
  auto it = nodes_.find(node_id);
  if (it == nodes_.end()) {
    return nullptr;
  }
  
  return it->second;
}

void InMemoryServiceRegistry::Subscribe(const std::string& service_name,
                                        std::function<void(const std::vector<ServiceNode>&)> callback) {
  std::unique_lock<std::shared_mutex> lock(callback_mutex_);
  callbacks_[service_name].push_back(callback);
  
  // 立即通知当前状态
  auto nodes = Discover(service_name);
  callback(nodes);
  
  spdlog::info("Subscribed to service: {}", service_name);
}

void InMemoryServiceRegistry::CleanupExpiredNodes(int timeout_seconds) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  
  auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  
  int64_t timeout_ms = timeout_seconds * 1000;
  std::vector<std::string> expired_nodes;
  
  for (const auto& [node_id, node] : nodes_) {
    if (now - node->last_heartbeat_time > timeout_ms) {
      expired_nodes.push_back(node_id);
    }
  }
  
  for (const auto& node_id : expired_nodes) {
    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) {
      std::string service_name = it->second->service_name;
      nodes_.erase(it);
      
      // 从服务索引中移除
      auto service_it = service_index_.find(service_name);
      if (service_it != service_index_.end()) {
        auto& node_ids = service_it->second;
        auto node_it = std::find(node_ids.begin(), node_ids.end(), node_id);
        if (node_it != node_ids.end()) {
          node_ids.erase(node_it);
        }
        if (node_ids.empty()) {
          service_index_.erase(service_it);
        }
      }
      
      spdlog::info("Expired node removed: {}", node_id);
    }
  }
}

bool InMemoryServiceRegistry::MarkNodeUnhealthy(const std::string& node_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  
  auto it = nodes_.find(node_id);
  if (it == nodes_.end()) {
    return false;
  }
  
  it->second->is_healthy = false;
  it->second->consecutive_failures++;
  spdlog::warn("Node marked as unhealthy: {}, failures: {}",
               node_id, it->second->consecutive_failures);
  
  return true;
}

std::vector<std::string> InMemoryServiceRegistry::GetAllServiceNames() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  
  std::vector<std::string> service_names;
  for (const auto& [service_name, node_ids] : service_index_) {
    service_names.push_back(service_name);
  }
  
  return service_names;
}

// ==================== HealthChecker 实现 ====================

HealthChecker::HealthChecker(const Config& config) : config_(config) {
}

HealthChecker::~HealthChecker() {
  Stop();
  if (check_thread_.joinable()) {
    check_thread_.join();
  }
}

void HealthChecker::Start(std::shared_ptr<ServiceRegistry> registry) {
  if (is_running_.load()) {
    return;
  }
  
  registry_ = registry;
  is_running_.store(true);
  check_thread_ = std::thread([this]() { CheckLoop(); });
  
  spdlog::info("Health checker started, interval: {}ms", config_.check_interval_ms);
}

void HealthChecker::Stop() {
  is_running_.store(false);
}

void HealthChecker::AddTarget(const std::string& node_id,
                              std::function<bool()> health_check_func) {
  std::lock_guard<std::mutex> lock(targets_mutex_);
  targets_[node_id] = health_check_func;
  spdlog::info("Health check target added: {}", node_id);
}

void HealthChecker::RemoveTarget(const std::string& node_id) {
  std::lock_guard<std::mutex> lock(targets_mutex_);
  targets_.erase(node_id);
  spdlog::info("Health check target removed: {}", node_id);
}

void HealthChecker::CheckLoop() {
  while (is_running_.load()) {
    {
      std::lock_guard<std::mutex> lock(targets_mutex_);
      for (const auto& [node_id, check_func] : targets_) {
        try {
          bool healthy = check_func();
          if (healthy) {
            registry_->Heartbeat(node_id);
          } else {
            registry_->MarkNodeUnhealthy(node_id);
          }
        } catch (const std::exception& e) {
          spdlog::error("Health check failed for {}: {}", node_id, e.what());
          registry_->MarkNodeUnhealthy(node_id);
        }
      }
    }
    
    // 等待下一次检查
    std::this_thread::sleep_for(
        std::chrono::milliseconds(config_.check_interval_ms));
  }
  
  spdlog::info("Health checker stopped");
}

// ==================== LoadBalancer 实现 ====================

std::shared_ptr<ServiceNode> RoundRobinLoadBalancer::Select(
    const std::string& service_name,
    const std::vector<ServiceNode>& nodes) {
  if (nodes.empty()) {
    return nullptr;
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = counters_.find(service_name);
  if (it == counters_.end()) {
    counters_[service_name] = 0;
  }
  
  size_t index = counters_[service_name] % nodes.size();
  counters_[service_name]++;
  
  return std::make_shared<ServiceNode>(nodes[index]);
}

void RoundRobinLoadBalancer::ReportResult(const std::string& /*node_id*/, 
                                          bool /*success*/, 
                                          int64_t /*latency_ms*/) {
  // 轮询负载均衡器不需要记录结果
}

std::shared_ptr<ServiceNode> WeightedRoundRobinLoadBalancer::Select(
    const std::string& /*service_name*/,
    const std::vector<ServiceNode>& nodes) {
  if (nodes.empty()) {
    return nullptr;
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  int total_weight = 0;
  int best_weight = -1;
  size_t best_index = 0;
  
  // 计算总权重
  for (size_t i = 0; i < nodes.size(); ++i) {
    total_weight += nodes[i].weight;
  }
  
  // 选择最佳节点
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    auto it = node_states_.find(node.node_id);
    if (it == node_states_.end()) {
      node_states_[node.node_id] = {0, node.weight};
    }
    
    auto& state = node_states_[node.node_id];
    state.current_weight += state.effective_weight;
    
    if (state.current_weight > best_weight) {
      best_weight = state.current_weight;
      best_index = i;
    }
  }
  
  // 调整权重
  auto& best_node = nodes[best_index];
  auto& best_state = node_states_[best_node.node_id];
  best_state.current_weight -= total_weight;
  
  return std::make_shared<ServiceNode>(best_node);
}

void WeightedRoundRobinLoadBalancer::ReportResult(const std::string& node_id, 
                                                  bool success, 
                                                  int64_t /*latency_ms*/) {
  // 可根据结果调整有效权重
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = node_states_.find(node_id);
  if (it != node_states_.end()) {
    if (!success) {
      // 失败时降低有效权重
      it->second.effective_weight = std::max(1, it->second.effective_weight - 1);
    } else {
      // 成功时恢复有效权重
      // 这里可以根据延迟动态调整
    }
  }
}

std::shared_ptr<ServiceNode> LeastConnectionsLoadBalancer::Select(
    const std::string& /*service_name*/,
    const std::vector<ServiceNode>& nodes) {
  if (nodes.empty()) {
    return nullptr;
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  // 找到连接数最少的节点
  size_t best_index = 0;
  int min_connections = nodes[0].active_requests;
  
  for (size_t i = 1; i < nodes.size(); ++i) {
    if (nodes[i].active_requests < min_connections) {
      min_connections = nodes[i].active_requests;
      best_index = i;
    }
  }
  
  // 增加连接计数
  connection_counts_[nodes[best_index].node_id]++;
  
  return std::make_shared<ServiceNode>(nodes[best_index]);
}

void LeastConnectionsLoadBalancer::ReportResult(const std::string& node_id, 
                                                bool /*success*/, 
                                                int64_t /*latency_ms*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = connection_counts_.find(node_id);
  if (it != connection_counts_.end() && it->second > 0) {
    it->second--;
  }
}

// ==================== ServiceGovernance 实现 ====================

ServiceGovernance& ServiceGovernance::GetInstance() {
  static ServiceGovernance instance;
  return instance;
}

bool ServiceGovernance::Initialize(std::shared_ptr<ServiceRegistry> registry,
                                   std::shared_ptr<LoadBalancer> load_balancer) {
  if (initialized_.load()) {
    return true;
  }
  
  registry_ = registry;
  load_balancer_ = load_balancer;
  
  // 创建健康检查器
  HealthChecker::Config config;
  health_checker_ = std::make_unique<HealthChecker>(config);
  health_checker_->Start(registry);
  
  initialized_.store(true);
  spdlog::info("Service governance initialized");
  
  return true;
}

bool ServiceGovernance::RegisterSelf(const std::string& node_id,
                                     const std::string& service_name,
                                     const std::string& host,
                                     int port,
                                     const std::unordered_map<std::string, std::string>& metadata) {
  if (!initialized_.load()) {
    spdlog::error("Service governance not initialized");
    return false;
  }
  
  ServiceNode node;
  node.node_id = node_id;
  node.service_name = service_name;
  node.host = host;
  node.port = port;
  node.version = "1.0.0";
  node.metadata = metadata;
  
  bool success = registry_->Register(node);
  if (success) {
    std::lock_guard<std::mutex> lock(self_mutex_);
    self_node_id_ = node_id;
    self_service_name_ = service_name;
    
    // 添加健康检查
    health_checker_->AddTarget(node_id, []() {
      // 健康检查逻辑：检查服务是否正常运行
      // 这里简化处理，实际应检查gRPC服务器状态
      return true;
    });
    
    spdlog::info("Self registered as node: {}", node_id);
  }
  
  return success;
}

bool ServiceGovernance::DeregisterSelf() {
  if (!initialized_.load()) {
    return false;
  }
  
  std::lock_guard<std::mutex> lock(self_mutex_);
  if (self_node_id_.empty()) {
    return false;
  }
  
  bool success = registry_->Deregister(self_node_id_);
  if (success) {
    health_checker_->RemoveTarget(self_node_id_);
    self_node_id_.clear();
    self_service_name_.clear();
    
    spdlog::info("Self deregistered");
  }
  
  return success;
}

std::vector<ServiceNode> ServiceGovernance::DiscoverService(const std::string& service_name) {
  if (!initialized_.load()) {
    return {};
  }
  
  return registry_->Discover(service_name);
}

std::shared_ptr<ServiceNode> ServiceGovernance::SelectNode(const std::string& service_name) {
  if (!initialized_.load()) {
    return nullptr;
  }
  
  auto nodes = registry_->Discover(service_name);
  if (nodes.empty()) {
    spdlog::warn("No healthy nodes found for service: {}", service_name);
    return nullptr;
  }
  
  return load_balancer_->Select(service_name, nodes);
}

void ServiceGovernance::ReportCallResult(const std::string& node_id, 
                                         bool success, 
                                         int64_t latency_ms) {
  if (!initialized_.load()) {
    return;
  }
  
  load_balancer_->ReportResult(node_id, success, latency_ms);
}

void ServiceGovernance::UpdateSelfStatus(float cpu_usage, float memory_usage, 
                                         int active_requests, float avg_latency_ms) {
  if (!initialized_.load()) {
    return;
  }
  
  std::lock_guard<std::mutex> lock(self_mutex_);
  if (self_node_id_.empty()) {
    return;
  }
  
  auto node = registry_->GetNode(self_node_id_);
  if (node) {
    node->cpu_usage = cpu_usage;
    node->memory_usage = memory_usage;
    node->active_requests = active_requests;
    node->avg_latency_ms = avg_latency_ms;
    
    registry_->UpdateNode(*node);
    registry_->Heartbeat(self_node_id_);
  }
}

}  // namespace distributed_inference
