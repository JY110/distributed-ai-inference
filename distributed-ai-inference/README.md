# 分布式AI推理服务

## 项目介绍

一个基于C++17、gRPC、ONNX Runtime构建的高性能分布式AI推理服务，支持高可用、高并发的模型推理能力。本项目采用分层架构设计，实现了完整的AI模型分布式推理解决方案，支持Docker一键部署，具备生产环境部署能力。

### 核心特性

- **高可用设计**：多节点部署，Nginx负载均衡，故障自动剔除
- **高性能**：ONNX Runtime引擎，批处理优化，内存池复用，多线程预处理
- **扩展性**：支持多种AI模型，预留TensorRT扩展接口
- **易用性**：Docker一键部署，标准化gRPC接口
- **监控**：完整的健康检查和服务状态监控
- **容错**：错误处理、异常捕获、全链路日志埋点

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| C++ | C++17 | 核心编程语言 |
| gRPC | 1.46+ | 服务间通信 |
| Protobuf | 3.19+ | 接口定义 |
| ONNX Runtime | 1.14+ | AI推理引擎 |
| OpenCV | 4.5+ | 图片预处理 |
| spdlog | 1.9+ | 结构化日志 |
| Nginx | 1.20+ | gRPC负载均衡 |
| Docker | 20.10+ | 容器化部署 |
| CMake | 3.20+ | 构建系统 |

## 项目结构

```
distributed-ai-inference/
├── src/                  # 源代码
│   ├── core/             # 核心推理模块
│   │   └── onnx_runtime_engine.cpp
│   ├── service/          # gRPC服务模块
│   │   └── inference_service_impl.cpp
│   ├── governance/       # 服务治理模块
│   │   └── service_registry.cpp
│   ├── proto/            # Protobuf定义
│   │   └── inference_service.proto
│   └── main.cpp          # 主程序
├── include/              # 头文件
│   ├── core/             # 核心模块头文件
│   │   └── onnx_runtime_engine.h
│   ├── service/          # 服务模块头文件
│   │   └── inference_service_impl.h
│   ├── governance/       # 治理模块头文件
│   │   └── service_registry.h
│   └── utils/            # 工具类头文件
│       ├── batch_processor.h
│       └── memory_pool.h
├── client/               # 客户端代码
│   └── client.cpp        # 测试客户端
├── models/               # 模型文件
│   └── resnet50-v1-12.onnx
├── nginx/                # Nginx配置
│   └── nginx.conf
├── build/                # 编译输出目录
│   ├── bin/              # 可执行文件
│   │   ├── inference-server
│   │   └── client
│   └── src/proto/        # 生成的gRPC代码
├── CMakeLists.txt        # CMake构建文件
├── Dockerfile            # Docker镜像构建文件
├── docker-compose.yml    # 多节点部署编排文件
└── README.md             # 项目文档
```

## 核心功能

### 1. 基础服务功能
- ✅ gRPC服务器启动和监听（默认端口50051）
- ✅ 健康检查接口
- ✅ 服务状态查询（运行时间、请求数、资源使用率等）
- ✅ 服务版本管理
- ✅ 请求ID生成和追踪

### 2. 模型管理功能
- ✅ 模型加载（支持ONNX格式）
- ✅ 模型注册和版本管理
- ✅ 模型卸载
- ✅ 模型状态查询
- ✅ 支持多个模型同时加载

### 3. 推理功能
- ✅ 单张图片推理
- ✅ 批量图片推理
- ✅ 推理结果返回（包含检测对象、置信度、位置等）
- ✅ 推理性能统计（推理时间、总时间）
- ✅ 支持自定义输入尺寸

### 4. 服务治理功能
- ✅ 服务注册与发现
- ✅ 健康检查机制
- ✅ 节点状态管理
- ✅ 节点标记为不健康
- ✅ 服务心跳检测

### 5. 性能优化功能
- ✅ 请求批处理（BatchProcessor）
- ✅ 内存池管理（MemoryPool）
- ✅ 多线程预处理
- ✅ 异步推理处理
- ✅ 并发请求处理

### 6. 容错与监控功能
- ✅ 错误处理和异常捕获
- ✅ 详细的日志记录（使用spdlog）
- ✅ 请求失败统计
- ✅ 性能指标收集
- ✅ 全链路日志埋点

### 7. 客户端功能
- ✅ gRPC客户端连接
- ✅ 健康检查测试
- ✅ 服务状态查询
- ✅ 模型加载测试
- ✅ 单张图片推理测试
- ✅ 批量推理测试

### 8. 部署支持
- ✅ Docker镜像构建
- ✅ Docker Compose编排
- ✅ Nginx负载均衡配置
- ✅ 多节点部署支持

## 性能指标

基于实际测试结果（ResNet50模型）：

| 指标 | 数值 |
|------|------|
| 模型加载时间 | ~348ms |
| 单张图片推理时间 | ~21ms |
| 批量推理（2张） | ~42ms |
| 并发处理能力 | 支持多客户端同时连接 |
| 内存使用 | 低内存占用，支持内存池复用 |

## 环境准备

### 依赖安装 (Ubuntu 22.04)

```bash
# 基础依赖
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    libgrpc++-dev \
    libopencv-dev \
    libspdlog-dev \
    wget \
    curl \
    git

# 安装ONNX Runtime
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz
tar -xzf onnxruntime-linux-x64-1.14.1.tgz
cd onnxruntime-linux-x64-1.14.1
cp -r include/* /usr/local/include/
cp -r lib/* /usr/local/lib/
ldconfig
```

### 模型准备

将你的ONNX模型文件放入 `models/` 目录，例如：
- YOLOv5 模型：`models/yolov5s.onnx`
- ResNet 模型：`models/resnet50-v1-12.onnx`

下载示例模型：
```bash
cd models
wget https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50-v1-12/resnet50-v1-12.onnx
```

## 编译步骤

### 本地编译

```bash
# 创建构建目录
mkdir -p build && cd build

# 配置项目
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 编译成功后会生成以下文件
# build/bin/inference-server  # 服务端可执行文件
# build/bin/client            # 客户端可执行文件
```

### 构建Docker镜像

```bash
docker build -t distributed-ai-inference .
```

## 部署方法

### 单节点部署

```bash
# 本地运行
cd build
./bin/inference-server

# 或使用Docker
docker run -d --name inference-server \
    -p 50051:50051 \
    -v $(pwd)/models:/models:ro \
    distributed-ai-inference
```

### 多节点分布式部署

使用 `docker-compose` 一键部署：

```bash
docker-compose up -d
```

这将启动：
- 2个推理服务节点（`inference-server-1` 和 `inference-server-2`）
- 1个Nginx负载均衡器（端口50051）
- 1个Redis缓存服务（可选）

### 扩展节点

编辑 `docker-compose.yml` 文件，添加更多推理节点：

```yaml
inference-server-3:
  build: .
  ports:
    - "50054:50051"
  volumes:
    - ./models:/models:ro
  restart: always
  networks:
    - inference-network
```

然后更新 Nginx 配置文件 `nginx/nginx.conf`，添加新节点到上游服务器列表。

## 测试用例

### 启动服务端

```bash
cd build
./bin/inference-server
```

**预期输出：**
```
[2026-04-02 XX:XX:XX.XXX] [info] Starting distributed AI inference service...
[2026-04-02 XX:XX:XX.XXX] [info] InferenceServiceImpl created, service_id: inference-service-xxxxx
[2026-04-02 XX:XX:XX.XXX] [info] gRPC server started on 0.0.0.0:50051
[2026-04-02 XX:XX:XX.XXX] [info] Inference service started successfully
[2026-04-02 XX:XX:XX.XXX] [info] Listening on: 0.0.0.0:50051
```

### 客户端测试

```bash
# 测试基础功能（健康检查、服务状态）
cd build
./bin/client localhost:50051

# 测试模型加载和推理
./bin/client localhost:50051 /path/to/image.jpg resnet50
```

**预期输出：**
```
[2026-04-02 XX:XX:XX.XXX] [info] Connected to server: localhost:50051
[2026-04-02 XX:XX:XX.XXX] [info] Testing health check
[2026-04-02 XX:XX:XX.XXX] [info] Health check successful
[2026-04-02 XX:XX:XX.XXX] [info] Service ID: inference-service-xxxxx
[2026-04-02 XX:XX:XX.XXX] [info] Status: 2
[2026-04-02 XX:XX:XX.XXX] [info] Version: 1.0.0

[2026-04-02 XX:XX:XX.XXX] [info] Testing service status
[2026-04-02 XX:XX:XX.XXX] [info] Service status successful
[2026-04-02 XX:XX:XX.XXX] [info] Uptime: 15 seconds
[2026-04-02 XX:XX:XX.XXX] [info] Total requests: 0
[2026-04-02 XX:XX:XX.XXX] [info] Active requests: 0

[2026-04-02 XX:XX:XX.XXX] [info] Testing load model: resnet50
[2026-04-02 XX:XX:XX.XXX] [info] Load model successful
[2026-04-02 XX:XX:XX.XXX] [info] Model: resnet50 vlatest
[2026-04-02 XX:XX:XX.XXX] [info] Load time: 348ms

[2026-04-02 XX:XX:XX.XXX] [info] Testing single image inference
[2026-04-02 XX:XX:XX.XXX] [info] Single image inference successful
[2026-04-02 XX:XX:XX.XXX] [info] Inference time: 21ms
[2026-04-02 XX:XX:XX.XXX] [info] Total time: 23ms
[2026-04-02 XX:XX:XX.XXX] [info] Detected objects: 0

[2026-04-02 XX:XX:XX.XXX] [info] Testing batch inference
[2026-04-02 XX:XX:XX.XXX] [info] Batch inference successful
[2026-04-02 XX:XX:XX.XXX] [info] Batch size: 2
[2026-04-02 XX:XX:XX.XXX] [info] Total time: 42ms
[2026-04-02 XX:XX:XX.XXX] [info] Success count: 2
[2026-04-02 XX:XX:XX.XXX] [info] Failed count: 0
```

### API测试

使用 gRPCurl 工具测试API：

```bash
# 安装 gRPCurl
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# 列出所有服务
grpcurl -plaintext localhost:50051 list

# 健康检查
grpcurl -plaintext localhost:50051 distributed_inference.InferenceService/HealthCheck

# 服务状态
grpcurl -plaintext localhost:50051 distributed_inference.InferenceService/GetServiceStatus
```

## API文档

### 主要RPC方法

| 方法 | 描述 | 请求 | 响应 |
|------|------|------|------|
| `Predict` | 单张图片推理 | `PredictRequest` | `PredictResponse` |
| `BatchPredict` | 批量图片推理 | `BatchPredictRequest` | `BatchPredictResponse` |
| `HealthCheck` | 健康检查 | `HealthCheckRequest` | `HealthCheckResponse` |
| `GetServiceStatus` | 获取服务状态 | `ServiceStatusRequest` | `ServiceStatusResponse` |
| `LoadModel` | 加载模型 | `LoadModelRequest` | `LoadModelResponse` |
| `UnloadModel` | 卸载模型 | `UnloadModelRequest` | `UnloadModelResponse` |

### 输入输出示例

#### 单张图片推理

**请求：**
```json
{
  "request_id": "test-123",
  "model_name": "yolov5s",
  "model_version": "latest",
  "task_type": "TASK_TYPE_DETECTION",
  "input_type": "INPUT_TYPE_IMAGE",
  "image": {
    "image_data": "<base64-encoded-image>",
    "format": "IMAGE_FORMAT_JPEG"
  }
}
```

**响应：**
```json
{
  "request_id": "test-123",
  "success": true,
  "inference_time_ms": 25,
  "model_name": "yolov5s",
  "model_version": "latest",
  "detection": {
    "boxes": [
      {
        "x": 100,
        "y": 150,
        "width": 50,
        "height": 60,
        "confidence": 0.92,
        "class_id": 0,
        "class_name": "person"
      }
    ],
    "image_width": 640,
    "image_height": 480
  }
}
```

## 性能优化

### 批处理优化
- 启用批处理：设置 `max_batch_size` 和 `batch_timeout_ms`
- 推荐批处理大小：8-32（根据模型和硬件调整）

### 内存优化
- 使用内存池：减少内存分配开销
- 预处理并行：使用多线程预处理
- 模型缓存：避免重复加载模型

### 负载均衡策略
- 轮询：适用于同构节点
- 加权轮询：根据节点性能调整权重
- 最少连接：适用于长连接场景

## 监控与日志

### 日志配置
- 日志级别：`info`（生产环境），`debug`（开发环境）
- 日志格式：`[时间戳] [级别] 消息`
- 日志内容：服务启动、请求处理、错误信息、性能指标

### 监控指标
- 服务状态：`GetServiceStatus` API
- 健康检查：`HealthCheck` API
- 推理性能：延迟、吞吐量
- 资源使用：CPU、内存

## 使用场景

1. **图像分类**：使用ResNet等模型进行图像分类
2. **目标检测**：支持YOLO等检测模型
3. **批量处理**：高效处理大量图片推理请求
4. **高并发服务**：支持多客户端同时访问
5. **分布式部署**：支持多节点负载均衡

## 常见问题

### 1. 模型加载失败

**原因：**
- 模型文件路径错误
- 模型格式不兼容
- 缺少依赖库

**解决：**
- 检查模型路径和文件权限
- 确保使用ONNX格式模型
- 安装完整的ONNX Runtime依赖

### 2. 推理速度慢

**原因：**
- 批处理大小不合适
- 内存池配置不当
- 硬件资源不足

**解决：**
- 调整批处理参数
- 增加内存池大小
- 部署更多推理节点

### 3. 服务不可用

**原因：**
- 端口被占用
- 网络连接问题
- 节点故障

**解决：**
- 检查端口占用情况
- 验证网络连接
- 查看Nginx和服务日志

### 4. 客户端连接超时

**原因：**
- 服务端未启动
- 防火墙阻止连接
- 网络配置问题

**解决：**
- 确认服务端已启动
- 检查防火墙设置
- 验证网络配置

## 扩展开发

### 添加新模型

1. 将ONNX模型文件放入 `models/` 目录
2. 使用 `LoadModel` API加载模型
3. 配置模型预处理参数

### 扩展推理引擎

预留了TensorRT扩展接口，可通过以下步骤集成：
1. 实现 `TensorRTEngine` 类（参考 `OnnxRuntimeEngine`）
2. 修改 `ModelManager` 支持多种引擎
3. 更新配置文件支持引擎选择

## 项目特点

1. **分层架构**：清晰的代码结构，易于维护和扩展
2. **高可用性**：支持多节点部署和故障转移
3. **高性能**：批处理、内存池、多线程优化
4. **可扩展性**：支持添加新的模型和功能
5. **生产就绪**：完整的日志、监控和错误处理

## 技术架构

### 分层设计

1. **客户端层**：提供gRPC客户端接口
2. **网关负载均衡层**：Nginx实现负载均衡
3. **推理服务节点层**：多个推理服务节点
4. **模型存储层**：模型文件存储和管理

### 核心模块

1. **接口模块**：gRPC推理接口定义
2. **推理核心模块**：ONNX Runtime封装
3. **服务治理模块**：节点健康检查、服务注册与发现
4. **性能优化模块**：请求批处理、内存池复用、多线程预处理并行
5. **容错与监控模块**：错误处理、异常捕获、全链路日志埋点

## 许可证

MIT License

## 联系信息

- 项目地址：https://github.com/yourusername/distributed-ai-inference
- 维护者：Your Name <your.email@example.com>
