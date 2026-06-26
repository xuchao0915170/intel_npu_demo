"""多模型流水线示例：人脸检测 + 分类。

演示如何组合多个模型完成复杂任务，例如：
1. 用人脸检测模型找出所有人脸
2. 对每个人脸区域裁剪
3. 用分类模型分析人脸特征（这里用 ImageNet 模型演示，实际可换成年龄/表情分类器）

这种流水线架构常用于智能监控、客流分析等场景。

用法:
    python examples/pipeline_face_classify.py --image 你的图片.jpg
    python examples/pipeline_face_classify.py --image test.jpg --device NPU --conf 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from npu_engine import NPUEngine
from tasks.classification import ClassificationTask
from tasks.face_detection import FaceDetectionTask


def load_labels(path: str) -> list[str] | None:
    if Path(path).exists():
        return Path(path).read_text(encoding="utf-8").splitlines()
    return None


def crop_face(image_bgr: np.ndarray, box: tuple[int, int, int, int],
              margin: float = 0.2) -> np.ndarray:
    """裁剪人脸区域，加一点边距避免太紧。"""
    x1, y1, x2, y2 = box
    h, w = image_bgr.shape[:2]

    # 加边距
    width, height = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - width * margin))
    y1 = max(0, int(y1 - height * margin))
    x2 = min(w, int(x2 + width * margin))
    y2 = min(h, int(y2 + height * margin))

    return image_bgr[y1:y2, x1:x2]


def main() -> None:
    parser = argparse.ArgumentParser(description="人脸检测 + 分类流水线")
    parser.add_argument("--image", required=True, help="输入图片")
    parser.add_argument("--device", default="NPU", help="设备")
    parser.add_argument("--conf", type=float, default=0.5, help="人脸检测置信度")
    parser.add_argument("--face-model", default="models/face-detection-0200.xml")
    parser.add_argument("--classify-model", default="models/resnet50.xml")
    parser.add_argument("--labels", default="models/imagenet_classes.txt")
    parser.add_argument("--output", help="可选：保存标注图到此路径")
    args = parser.parse_args()

    print("=" * 60)
    print("多模型流水线：人脸检测 → 区域分类")
    print("=" * 60)

    # 阶段 1：人脸检测
    print("\n[Stage 1] 人脸检测")
    face_engine = NPUEngine(args.face_model, device=args.device, input_shape=(1, 3, 256, 256))
    print(f"  引擎: {face_engine}")
    face_task = FaceDetectionTask(face_engine, input_size=256, conf_threshold=args.conf)

    faces = face_task.run(args.image)
    print(f"  检测到 {len(faces)} 张人脸")

    if not faces:
        print("  未检测到人脸，退出")
        return

    # 阶段 2：对每个人脸做分类（演示用，实际应换成专门的人脸属性模型）
    print("\n[Stage 2] 人脸区域分类（ImageNet 演示）")
    classify_engine = NPUEngine(args.classify_model, device=args.device,
                                 input_shape=(1, 3, 224, 224))
    print(f"  引擎: {classify_engine}")
    classify_task = ClassificationTask(classify_engine,
                                       labels=load_labels(args.labels), image_size=224)

    # 读原图用于裁剪
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"无法读取图片: {args.image}")

    results = []
    for i, face in enumerate(faces, 1):
        # 裁剪人脸区域
        face_crop = crop_face(image_bgr, face["box"])

        # 保存临时图片供分类任务读取（实际生产中可改成直接传 ndarray）
        temp_path = f"_temp_face_{i}.jpg"
        cv2.imwrite(temp_path, face_crop)

        # 分类
        preds = classify_task.run(temp_path)[:3]  # Top-3
        Path(temp_path).unlink()  # 删除临时文件

        results.append({
            "face_id": i,
            "confidence": face["confidence"],
            "box": face["box"],
            "predictions": preds,
        })

        print(f"\n  人脸 #{i} (置信度 {face['confidence']:.2f})")
        print(f"    位置: {face['box']}")
        print(f"    分类 Top-3:")
        for rank, (idx, label, prob) in enumerate(preds, 1):
            print(f"      {rank}. {label:<30} {prob*100:.2f}%")

    # 可选：保存可视化结果
    if args.output:
        for r in results:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在框上方显示 Top-1 类别
            top1_label = r["predictions"][0][1]
            cv2.putText(image_bgr, f"#{r['face_id']}: {top1_label[:15]}",
                       (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)

        cv2.imwrite(args.output, image_bgr)
        print(f"\n已保存标注图: {args.output}")

    print("\n" + "=" * 60)
    print("流水线完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
