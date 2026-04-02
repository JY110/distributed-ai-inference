cmake_minimum_required(VERSION 3.20)
project(distributed-ai-inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 查找依赖
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

# 查找Protobuf
find_package(Protobuf REQUIRED)

# 查找gRPC
find_package(gRPC REQUIRED)

# 查找ONNX Runtime
find_library(ONNX_RUNTIME_LIBRARY onnxruntime HINTS /usr/lib/x86_64-linux-gnu)
find_path(ONNX_RUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS /usr/include)

if(NOT ONNX_RUNTIME_LIBRARY OR NOT ONNX_RUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime not found")
endif()

# 生成Proto文件
source_group("Proto Files" FILES src/proto/inference_service.proto)

# 生成Protobuf和gRPC代码
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
if(NOT GRPC_CPP_PLUGIN)
    message(FATAL_ERROR "grpc_cpp_plugin not found")
endif()

# 直接使用命令行生成代码，避免CMake路径问题
add_custom_target(generate_proto
    COMMAND protoc
    --cpp_out=${CMAKE_BINARY_DIR}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
    src/proto/inference_service.proto
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/proto/inference_service.proto
)

# 设置生成的文件
set(PROTO_SRCS ${CMAKE_BINARY_DIR}/inference_service.pb.cc)
set(PROTO_HDRS ${CMAKE_BINARY_DIR}/inference_service.pb.h)
set(GRPC_SRCS ${CMAKE_BINARY_DIR}/inference_service.grpc.pb.cc)
set(GRPC_HDRS ${CMAKE_BINARY_DIR}/inference_service.grpc.pb.h)

# 确保生成的文件被创建
execute_process(
    COMMAND protoc
    --cpp_out=${CMAKE_BINARY_DIR}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
    src/proto/inference_service.proto
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE PROTOC_RESULT
)

if(PROTOC_RESULT)
    message(FATAL_ERROR "protoc failed with result: ${PROTOC_RESULT}")
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/opencv4
    /usr/include/opencv4/opencv2
    ${spdlog_INCLUDE_DIRS}
    ${ONNX_RUNTIME_INCLUDE_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)

# 源文件
set(SOURCE_FILES
    # 核心模块
    src/core/onnx_runtime_engine.cpp
    
    # 治理模块
    src/governance/service_registry.cpp
    
    # 服务模块
    src/service/inference_service_impl.cpp
    
    # 主程序
    src/main.cpp
)

# 库文件
set(LIBRARY_FILES
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

# 构建服务端可执行文件
add_executable(inference-server ${SOURCE_FILES} ${LIBRARY_FILES})
target_link_libraries(inference-server
    ${OpenCV_LIBS}
    ${spdlog_LIBRARIES}
    ${ONNX_RUNTIME_LIBRARY}
    ${Protobuf_LIBRARIES}
    ${gRPC_LIBRARIES}
    grpc++
    grpc++_reflection
    gpr
    absl_synchronization
    fmt
    Threads::Threads
)

# 构建客户端可执行文件
add_executable(client client/client.cpp ${LIBRARY_FILES})
target_include_directories(client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/opencv4
    /usr/include/opencv4/opencv2
    ${spdlog_INCLUDE_DIRS}
    ${ONNX_RUNTIME_INCLUDE_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)
target_link_libraries(client
    ${OpenCV_LIBS}
    ${spdlog_LIBRARIES}
    ${ONNX_RUNTIME_LIBRARY}
    ${Protobuf_LIBRARIES}
    ${gRPC_LIBRARIES}
    grpc++
    gpr
    absl_synchronization
    fmt
    Threads::Threads
)

# 安装配置
install(TARGETS inference-server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 安装模型目录
install(DIRECTORY models/ DESTINATION models)

# 安装配置文件
install(DIRECTORY configs/ DESTINATION configs)

# 构建测试
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# 打印配置信息
message(STATUS "==================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "CMake Version: ${CMAKE_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNX_RUNTIME_LIBRARY}")
message(STATUS "gRPC Version: ${gRPC_VERSION}")
message(STATUS "Protobuf Version: ${Protobuf_VERSION}")
message(STATUS "==================================")
