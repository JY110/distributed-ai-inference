FROM ubuntu:22.04

LABEL maintainer="distributed-ai-inference"
LABEL version="1.0.0"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libprotobuf-dev \
    protobuf-compiler \
    libgrpc++-dev \
    grpc-tools \
    libopencv-dev \
    libspdlog-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装ONNX Runtime
RUN wget -O onnxruntime.deb https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.deb \
    && dpkg -i onnxruntime.deb \
    && rm onnxruntime.deb

# 创建工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 构建项目
RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install

# 创建模型目录
RUN mkdir -p /models

# 下载示例模型（可选）
RUN wget -O /models/yolov5s.onnx https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.onnx

# 暴露端口
EXPOSE 50051

# 启动服务
CMD ["/app/build/bin/inference-server"]
