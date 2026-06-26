# Intel NPU 推理框架 Demo (OpenVINO + Python)

一个**生产级、可扩展**的推理框架,演示如何用 [OpenVINO](https://docs.openvino.ai/)
在 **Intel NPU**（Meteor Lake / Core Ultra 及以上的集成 NPU）上做模型推理。
没有 NPU 的机器也能跑——引擎会自动回退到 GPU 或 CPU。

## 特性

- ✅ **设备自动回退** — NPU 不可用或编译失败时,自动降级到 GPU → CPU,开发机友好
- ✅ **静态 shape 处理** — 自动满足 NPU 的静态 shape 要求,无需手动 reshape
- ✅ **引擎与任务解耦** — 换模型/换任务只需改一个类,框架代码不动
- ✅ **三种完整应用** — 图像分类、人脸检测、实时摄像头,开箱即用

## 目录结构

```
npu_demo/
├── npu_engine/
│   ├── engine.py          # OpenVINO 推理引擎封装（设备回退、编译、测速）
│   └── base_task.py       # 任务抽象基类（预处理/后处理接口）
├── tasks/
│   ├── classification.py  # 图像分类任务（ImageNet 1000 类）
│   └── face_detection.py  # 人脸检测任务（支持单图+实时帧）
├── export_model.py        # 导出 torchvision 模型到 OpenVINO IR
├── main.py                # 命令行工具（分类/检测两种模式）
├── webcam_face.py         # 摄像头实时人脸检测
└── requirements.txt
```

## 环境准备

1. **安装 Python 依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **NPU 驱动**（仅真实 Intel NPU 需要）：
   - Windows 通常随显卡/系统更新自带
   - 在"设备管理器 → 神经处理器"确认能看到 `Intel(R) AI Boost`
   - 或手动下载：https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html

3. **确认 OpenVINO 能识别设备**：
   ```bash
   python main.py --list-devices
   ```
   输出应包含 `NPU -> Intel(R) AI Boost`（没有 NPU 也能用 GPU/CPU）。

## 快速开始

### 1. 图像分类（ImageNet 1000 类）

```bash
# 导出示例模型（ResNet50）
python export_model.py

# 单张图片分类
python main.py --task classify --model models/resnet50.xml --image 你的图片.jpg --device NPU

# 性能测试
python main.py --task classify --model models/resnet50.xml --benchmark --device NPU
```

**可选模型**（都是 ImageNet 预训练,精度从 ~70% 到 ~81%）：
```bash
python export_model.py --list               # 查看可选模型
python export_model.py --model resnet18     # 轻量、最快
python export_model.py --model resnet50     # 精度更高（推荐）
python export_model.py --model mobilenet_v3 # 移动端友好
python export_model.py --model efficientnet_b0
```

### 2. 人脸检测（单张图片）

```bash
# 模型会自动从 OpenVINO Model Zoo 下载（脚本已内置）
python main.py --task face --model models/face-detection-0200.xml --image 图片.jpg --device NPU

# 输出示例:
# 检测到 2 张人脸:
#   1. 置信度 0.98  框=(120,80)-(240,220)
#   2. 置信度 0.95  框=(300,100)-(400,250)
# 已保存标注图: faces_out.jpg
```

### 3. 摄像头实时人脸检测 ⭐

```bash
python webcam_face.py
```

会弹出窗口显示实时画面,左上角显示 `NPU  faces=1  3.0ms  330FPS`。  
**按 q 或 ESC 退出。**

常用参数：
```bash
python webcam_face.py --device NPU       # 默认用 NPU
python webcam_face.py --device CPU       # 对比 CPU 帧率
python webcam_face.py --camera 1         # 如有多个摄像头,换编号
python webcam_face.py --conf 0.6         # 调高置信度阈值
```

## 性能参考（Core Ultra 5 125H）

| 任务 | 模型 | NPU 延迟 | NPU 吞吐 | GPU 延迟 | CPU 延迟 |
|------|------|---------|---------|---------|---------|
| 图像分类 | ResNet50 | 4.64 ms | 215 FPS | 1.74 ms | 10.24 ms |
| 人脸检测 | face-detection-0200 | 2.81 ms | 355 FPS | - | - |
| 实时摄像头 | 同上 | ~3 ms | **330 FPS** | - | - |

**为什么 GPU 更快？**  
对 ResNet50 这类轻量模型,GPU 并行度优势明显。NPU 的优势在于:
- **低功耗**:多模型并发、长时运行时发热低
- **不占 GPU**:GPU 可留给游戏/渲染/训练
- **更大模型**:YOLO、分割网络等,NPU 优势更明显

## 关于 Intel NPU 的两个关键点

1. **静态 shape**：NPU 插件不支持动态输入尺寸。本框架在编译前自动用
   `input_shape` 把模型 reshape 成固定尺寸（例如 `1x3x224x224`）。
2. **精度提示**：NPU 上默认设置 `INFERENCE_PRECISION_HINT=f16`，通常更快、占用更小。

## 扩展到你的业务

### 方法 1：换模型

把你的 ONNX 或 OpenVINO IR 放进 `models/`,直接用:
```bash
python main.py --task classify --model models/你的模型.onnx --image test.jpg --device NPU
```

`NPUEngine` 同时支持 `.xml`（OpenVINO IR）和 `.onnx`。

### 方法 2：新增任务类

以目标检测（YOLO）为例:

```python
from npu_engine import BaseTask

class YOLOTask(BaseTask):
    def preprocess(self, image_path):
        # 你的预处理：读图、resize、归一化
        ...
        return tensor  # NCHW

    def postprocess(self, outputs):
        # 你的后处理：解析框、NMS、类别映射
        ...
        return results
```

然后在 `main.py` 里实例化 `YOLOTask` 即可,引擎代码不用改。

### 典型场景

| 场景 | 改造方式 |
|------|---------|
| **工业质检** | 换缺陷检测/分类模型 → 改 `postprocess` 输出"合格/不合格" |
| **智能监控** | 换 YOLO 检测人/车 → 加 NMS 后处理 → 多路流并发 |
| **医疗影像** | 换病灶分类/分割模型 → 预处理改 DICOM 读取 |
| **农业分级** | 换水果识别模型 → 传送带摄像头实时抓拍 |

## 模型来源

| 模型 | 来源 | 许可 |
|------|------|------|
| ResNet / MobileNet / EfficientNet | **torchvision** (PyTorch 官方) | BSD-3-Clause,可商用 |
| face-detection-0200 | **OpenVINO Model Zoo** (Intel 官方) | Apache 2.0,可商用 |

所有模型均为开源、在公开数据集上预训练的标准模型。

## 常见问题

### Q: NPU 编译失败 "Unsupported operation MaxPool"
**A:** NPU 驱动版本太旧。更新到最新版（32.0.100.3xxx+）：  
https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html

框架已内置编译失败时自动回退到 GPU/CPU,不影响开发。

### Q: 摄像头打不开或黑屏
**A:** 
- 换 `--camera 1` 或 `2`（笔记本前置摄像头可能不是 0 号）
- 关闭其他占用摄像头的程序（Teams、相机 App 等）

### Q: 第一帧很卡
**A:** NPU 首次编译模型有几秒延迟,属正常,之后就流畅了。

### Q: 想用 C++ 开发
**A:** 核心逻辑一致,用 OpenVINO C++ API。需要:
- Visual Studio 2019/2022 + MSVC
- OpenVINO Toolkit (Archive 版)
- CMake 3.16+

参考 [main.py](main.py) 的流程用 `ov::Core` / `ov::CompiledModel` / `ov::InferRequest` 实现。

## 项目结构设计

**核心思想**：推理引擎与任务逻辑分离
- `NPUEngine` — 只管"加载模型、选设备、跑推理",与具体任务无关
- `BaseTask` — 定义 `preprocess` / `postprocess` 接口
- 新增任务 — 继承 `BaseTask`,实现两个方法即可,引擎代码零修改

这种设计让框架既简单又可扩展,适合快速原型和生产部署。

## 许可

MIT License. 示例代码和框架可自由用于商业/非商业项目。  
预训练模型遵循各自的开源许可（BSD-3 / Apache 2.0）。

---

需要帮助？欢迎提 issue 或查看 OpenVINO 官方文档：https://docs.openvino.ai/
