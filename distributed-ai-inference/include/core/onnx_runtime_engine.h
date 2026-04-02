#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>

namespace distributed_inference {

/**
 * @brief 模型配置结构体
 * 存储模型的预处理和后处理配置参数
 */
struct ModelConfig {
  std::string model_name;           // 模型名称
  std::string model_version;        // 模型版本
  std::string model_path;           // 模型文件路径
  std::vector<int64_t> input_shape; // 输入张量形状
  std::vector<std::string> input_names;   // 输入节点名称
  std::vector<std::string> output_names;  // 输出节点名称
  
  // 预处理参数
  int target_width = 640;           // 目标宽度
  int target_height = 640;          // 目标高度
  bool keep_aspect_ratio = true;    // 是否保持宽高比
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};  // 归一化均值
  std::vector<float> std = {0.229f, 0.224f, 0.225f};   // 归一化标准差
  float scale = 1.0f / 255.0f;      // 缩放因子
  
  // 后处理参数
  float conf_threshold = 0.25f;     // 置信度阈值
  float nms_threshold = 0.45f;      // NMS阈值
  int num_classes = 80;             // 类别数量
  
  // 执行配置
  int intra_op_num_threads = 4;     // 算子内并行线程数
  int inter_op_num_threads = 4;     // 算子间并行线程数
  bool use_gpu = false;             // 是否使用GPU
  int gpu_device_id = 0;            // GPU设备ID
};

/**
 * @brief 推理结果结构体
 * 封装单张图片的推理输出
 */
struct InferenceResult {
  bool success = false;             // 推理是否成功
  std::string error_message;        // 错误信息
  int64_t inference_time_ms = 0;    // 推理耗时(毫秒)
  
  // 检测结果 (目标检测)
  struct Detection {
    float x, y, width, height;      // 边界框坐标
    float confidence;               // 置信度
    int class_id;                   // 类别ID
    std::string class_name;         // 类别名称
  };
  std::vector<Detection> detections;
  
  // 分类结果
  struct Classification {
    int class_id;                   // 类别ID
    std::string class_name;         // 类别名称
    float confidence;               // 置信度
  };
  std::vector<Classification> classifications;
  
  // 原始输出张量 (用于自定义后处理)
  std::vector<std::vector<float>> raw_outputs;
};

/**
 * @brief ONNX Runtime推理引擎类
 * 
 * 设计思路：
 * 1. 封装ONNX Runtime C++ API，提供统一的模型推理接口
 * 2. 支持模型热加载和动态切换
 * 3. 集成OpenCV实现图片预处理
 * 4. 线程安全设计，支持并发推理
 * 5. 支持CPU/GPU多种执行环境
 */
class OnnxRuntimeEngine {
 public:
  OnnxRuntimeEngine();
  ~OnnxRuntimeEngine();

  // 禁止拷贝，允许移动
  OnnxRuntimeEngine(const OnnxRuntimeEngine&) = delete;
  OnnxRuntimeEngine& operator=(const OnnxRuntimeEngine&) = delete;
  OnnxRuntimeEngine(OnnxRuntimeEngine&&) noexcept;
  OnnxRuntimeEngine& operator=(OnnxRuntimeEngine&&) noexcept;

  /**
   * @brief 初始化推理引擎
   * @param config 模型配置
   * @return 是否初始化成功
   */
  bool Initialize(const ModelConfig& config);

  /**
   * @brief 加载模型
   * @param model_path 模型文件路径
   * @param config 模型配置
   * @return 是否加载成功
   */
  bool LoadModel(const std::string& model_path, const ModelConfig& config);

  /**
   * @brief 卸载当前模型
   */
  void UnloadModel();

  /**
   * @brief 检查模型是否已加载
   */
  bool IsModelLoaded() const;

  /**
   * @brief 执行推理 (图片输入)
   * @param image OpenCV图片矩阵
   * @return 推理结果
   */
  InferenceResult Infer(const cv::Mat& image);

  /**
   * @brief 执行推理 (张量输入)
   * @param input_data 输入张量数据
   * @param shape 张量形状
   * @return 推理结果
   */
  InferenceResult Infer(std::vector<float>& input_data,
                        const std::vector<int64_t>& shape);

  /**
   * @brief 批量推理 (图片输入)
   * @param images 图片列表
   * @return 推理结果列表
   */
  std::vector<InferenceResult> BatchInfer(const std::vector<cv::Mat>& images);

  /**
   * @brief 获取模型信息
   */
  std::string GetModelInfo() const;

  /**
   * @brief 获取输入形状
   */
  std::vector<int64_t> GetInputShape() const;

  /**
   * @brief 获取输出形状
   */
  std::vector<std::vector<int64_t>> GetOutputShapes() const;

 private:
  /**
   * @brief 预处理图片
   * @param image 原始图片
   * @return 预处理后的张量数据
   */
  std::vector<float> Preprocess(const cv::Mat& image);

  /**
   * @brief 后处理 - 目标检测
   * @param outputs 模型原始输出
   * @param orig_width 原始图片宽度
   * @param orig_height 原始图片高度
   * @return 结构化检测结果
   */
  InferenceResult PostprocessDetection(
      const std::vector<Ort::Value>& outputs,
      int orig_width, int orig_height);

  /**
   * @brief 后处理 - 图像分类
   */
  InferenceResult PostprocessClassification(
      const std::vector<Ort::Value>& outputs);

  /**
   * @brief 执行NMS (非极大值抑制)
   */
  std::vector<int> Nms(const std::vector<InferenceResult::Detection>& boxes,
                       float nms_threshold);

  /**
   * @brief 创建Ort会话选项
   */
  Ort::SessionOptions CreateSessionOptions(const ModelConfig& config);

 private:
  // ONNX Runtime环境 (全局共享)
  static std::shared_ptr<Ort::Env> env_;
  static std::mutex env_mutex_;
  static int env_ref_count_;

  // 会话和内存分配器
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;

  // 模型配置
  ModelConfig config_;
  mutable std::mutex config_mutex_;

  // 输入输出信息缓存
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_names_ptr_;
  std::vector<const char*> output_names_ptr_;
  std::vector<int64_t> input_shape_;
  std::vector<std::vector<int64_t>> output_shapes_;

  // 推理统计
  int64_t inference_count_ = 0;
  int64_t total_inference_time_ms_ = 0;
  mutable std::mutex stats_mutex_;
};

/**
 * @brief 模型管理器类
 * 
 * 设计思路：
 * 1. 管理多个模型的加载和切换
 * 2. 支持模型热加载，无需重启服务
 * 3. 线程安全的模型访问
 */
class ModelManager {
 public:
  static ModelManager& GetInstance();

  /**
   * @brief 注册模型
   */
  bool RegisterModel(const std::string& model_name,
                     const std::string& model_version,
                     const ModelConfig& config);

  /**
   * @brief 加载模型到引擎
   */
  bool LoadModel(const std::string& model_name,
                 const std::string& model_version);

  /**
   * @brief 卸载模型
   */
  bool UnloadModel(const std::string& model_name,
                   const std::string& model_version);

  /**
   * @brief 获取模型引擎
   */
  std::shared_ptr<OnnxRuntimeEngine> GetEngine(
      const std::string& model_name,
      const std::string& model_version);

  /**
   * @brief 获取默认引擎
   */
  std::shared_ptr<OnnxRuntimeEngine> GetDefaultEngine();

  /**
   * @brief 列出所有已加载的模型
   */
  std::vector<std::pair<std::string, std::string>> ListLoadedModels();

  /**
   * @brief 获取模型状态
   */
  struct ModelStatus {
    std::string model_name;
    std::string model_version;
    bool is_loaded;
    int64_t inference_count;
    float avg_inference_time_ms;
  };
  std::vector<ModelStatus> GetAllModelStatus();

 private:
  ModelManager() = default;
  ~ModelManager() = default;

  // 模型引擎映射表
  std::unordered_map<std::string, 
                     std::unordered_map<std::string, 
                                        std::shared_ptr<OnnxRuntimeEngine>>> engines_;
  mutable std::shared_mutex engines_mutex_;

  // 模型配置映射表
  std::unordered_map<std::string, 
                     std::unordered_map<std::string, ModelConfig>> configs_;
  mutable std::shared_mutex configs_mutex_;

  // 默认模型
  std::string default_model_name_;
  std::string default_model_version_;
  mutable std::shared_mutex default_mutex_;
};

}  // namespace distributed_inference
