"""批量处理图片并保存分类结果到 CSV。

演示如何用推理引擎批量处理多张图片，适合数据集标注、质量检查等场景。

用法:
    python examples/batch_classify.py --input images/ --output results.csv
    python examples/batch_classify.py --input images/ --device GPU --model models/resnet50.xml
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from npu_engine import NPUEngine
from tasks.classification import ClassificationTask


def load_labels(path: str) -> list[str] | None:
    if Path(path).exists():
        return Path(path).read_text(encoding="utf-8").splitlines()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="批量图片分类")
    parser.add_argument("--input", required=True, help="输入目录或图片列表文件")
    parser.add_argument("--output", default="results.csv", help="输出 CSV 文件")
    parser.add_argument("--model", default="models/resnet50.xml", help="模型路径")
    parser.add_argument("--device", default="NPU", help="NPU/GPU/CPU")
    parser.add_argument("--labels", default="models/imagenet_classes.txt")
    parser.add_argument("--size", type=int, default=224, help="输入尺寸")
    parser.add_argument("--top", type=int, default=5, help="保存 Top-N 结果")
    args = parser.parse_args()

    # 收集图片路径
    input_path = Path(args.input)
    if input_path.is_dir():
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    elif input_path.is_file():
        images = [Path(line.strip()) for line in input_path.read_text().splitlines()]
    else:
        raise ValueError(f"输入路径不存在: {args.input}")

    if not images:
        raise ValueError(f"未找到图片: {args.input}")

    print(f"[batch] 找到 {len(images)} 张图片")

    # 初始化引擎和任务
    engine = NPUEngine(args.model, device=args.device,
                       input_shape=(1, 3, args.size, args.size))
    print(f"[batch] {engine}")

    task = ClassificationTask(engine, labels=load_labels(args.labels), image_size=args.size)

    # 批量推理
    results = []
    t0 = time.perf_counter()
    for i, img_path in enumerate(images, 1):
        try:
            preds = task.run(img_path)[:args.top]
            for rank, (idx, label, prob) in enumerate(preds, 1):
                results.append({
                    "image": img_path.name,
                    "rank": rank,
                    "class_id": idx,
                    "class_name": label,
                    "confidence": f"{prob:.4f}",
                })
            if i % 10 == 0:
                print(f"[batch] 已处理 {i}/{len(images)}")
        except Exception as e:
            print(f"[batch] {img_path.name} 失败: {e}")

    elapsed = time.perf_counter() - t0
    print(f"[batch] 完成 {len(images)} 张图片，耗时 {elapsed:.1f}s ({len(images)/elapsed:.1f} FPS)")

    # 保存 CSV
    if results:
        output = Path(args.output)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[batch] 结果已保存: {output} ({len(results)} 条记录)")


if __name__ == "__main__":
    main()
