"""命令行入口：在 Intel NPU（或回退设备）上做图像分类推理。

示例:
    # 1. 导出示例模型
    python export_model.py

    # 2. 查看可用设备
    python main.py --list-devices

    # 3. 推理一张图片
    python main.py --model models/resnet18.xml --image cat.jpg --device NPU

    # 4. 测速
    python main.py --model models/resnet18.xml --benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openvino as ov

from npu_engine import NPUEngine
from tasks.classification import ClassificationTask
from tasks.face_detection import FaceDetectionTask


def load_labels(path: str | None) -> list[str] | None:
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8").splitlines()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Intel NPU OpenVINO 推理 demo")
    parser.add_argument("--model", help="模型路径 (.xml 或 .onnx)")
    parser.add_argument("--image", help="待推理的图片路径")
    parser.add_argument("--device", default="NPU", help="目标设备: NPU/GPU/CPU/AUTO")
    parser.add_argument("--task", default="classify", choices=["classify", "face"],
                        help="任务类型: classify=图像分类, face=人脸检测")
    parser.add_argument("--labels", default="models/imagenet_classes.txt",
                        help="类别标签文件（仅分类任务）")
    parser.add_argument("--size", type=int, default=224, help="输入边长")
    parser.add_argument("--conf", type=float, default=0.5, help="人脸检测置信度阈值")
    parser.add_argument("--benchmark", action="store_true", help="跑测速而非单张推理")
    parser.add_argument("--list-devices", action="store_true", help="列出可用设备后退出")
    args = parser.parse_args()

    if args.list_devices:
        core = ov.Core()
        print("可用设备:")
        for d in core.available_devices:
            name = core.get_property(d, "FULL_DEVICE_NAME")
            print(f"  {d:6s} -> {name}")
        return

    if not args.model:
        parser.error("需要 --model 指定模型（或用 --list-devices）")

    # 人脸检测模型输入是 256，分类是 224
    size = 256 if args.task == "face" else args.size
    input_shape = (1, 3, size, size)
    engine = NPUEngine(args.model, device=args.device, input_shape=input_shape)
    print(f"[main] {engine}")

    if args.task == "face":
        task = FaceDetectionTask(engine, input_size=size, conf_threshold=args.conf)
    else:
        task = ClassificationTask(engine, labels=load_labels(args.labels), image_size=size)

    if args.benchmark:
        dummy = np.random.rand(*input_shape).astype(np.float32)
        avg_ms = engine.benchmark(dummy)
        print(f"[main] 设备 {engine.device} 平均单次推理: {avg_ms:.2f} ms "
              f"({1000 / avg_ms:.1f} FPS)")
        return

    if not args.image:
        parser.error("需要 --image 指定图片（或用 --benchmark）")

    results = task.run(args.image)

    if args.task == "face":
        print(f"\n检测到 {len(results)} 张人脸:")
        for i, f in enumerate(results, 1):
            x1, y1, x2, y2 = f["box"]
            print(f"  {i}. 置信度 {f['confidence']:.2f}  框=({x1},{y1})-({x2},{y2})")
        if results:
            out = task.draw(args.image, results)
            print(f"\n已保存标注图: {out}")
    else:
        print(f"\nTop-{len(results)} 预测结果:")
        for rank, (idx, label, prob) in enumerate(results, 1):
            print(f"  {rank}. [{idx:>4}] {label:<30} {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
