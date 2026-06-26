# 示例脚本

这个目录包含实用的示例脚本,演示如何在实际场景中使用 NPU 推理框架。

## 📋 batch_classify.py - 批量图片分类

批量处理多张图片并保存结果到 CSV,适合数据集标注、质量检查等场景。

**用法**:
```bash
# 处理整个目录
python examples/batch_classify.py --input images/ --output results.csv

# 指定设备和模型
python examples/batch_classify.py --input images/ --device GPU --model models/resnet50.xml

# 只保存 Top-3 结果
python examples/batch_classify.py --input images/ --top 3
```

**输出**: CSV 文件,每行包含 `image, rank, class_id, class_name, confidence`

---

## 📊 benchmark_compare.py - 性能对比测试

对比不同设备（NPU/GPU/CPU）和模型的推理性能,生成详细报告。

**用法**:
```bash
# 默认测试 resnet50 在所有设备上的性能
python examples/benchmark_compare.py

# 对比多个模型
python examples/benchmark_compare.py --models models/resnet18.xml models/resnet50.xml

# 只测试特定设备
python examples/benchmark_compare.py --devices NPU GPU

# 调整测试参数
python examples/benchmark_compare.py --iterations 100 --warmup 10
```

**输出示例**:
```
模型                 请求设备     实际设备     编译(ms)     平均(ms)     吞吐(FPS)
--------------------------------------------------------------------------------
resnet50             NPU        NPU        1234.5       4.64         215
resnet50             GPU        GPU        567.8        1.74         575
resnet50             CPU        CPU        89.3         10.24        98
```

---

## 🔗 pipeline_face_classify.py - 多模型流水线

演示如何组合多个模型完成复杂任务：人脸检测 → 裁剪人脸 → 分类分析。

这种流水线架构常用于智能监控、客流分析、人脸属性识别等场景。

**用法**:
```bash
# 基本使用
python examples/pipeline_face_classify.py --image 你的图片.jpg

# 保存可视化结果
python examples/pipeline_face_classify.py --image test.jpg --output result.jpg

# 调整检测阈值
python examples/pipeline_face_classify.py --image test.jpg --conf 0.7 --device GPU
```

**流程**:
1. 人脸检测模型找出所有人脸位置
2. 裁剪每个人脸区域（加适当边距）
3. 对每个人脸运行分类模型（演示用 ImageNet，实际可换成年龄/表情分类器）
4. 输出每个人脸的 Top-3 预测结果

**注**: 这里用 ImageNet 分类模型演示流水线架构,实际生产中应替换为专门的人脸属性模型（年龄、性别、表情等）。

---

## 扩展建议

基于这些示例,你可以：

- **batch_classify.py** → 改成批量缺陷检测、商品识别等
- **benchmark_compare.py** → 加入内存占用、功耗监控
- **pipeline_face_classify.py** → 扩展成检测→跟踪→识别完整流水线

所有脚本都使用框架的核心 API,换模型或任务只需修改几行配置。
