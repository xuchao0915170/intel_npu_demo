"""导出 torchvision 分类模型到 OpenVINO IR。

支持多个常见模型，方便对比精度/速度。生产中你也可以直接把自己的
ONNX/IR 模型放到 models/ 目录，跳过本脚本。

用法:
    python export_model.py                       # 默认导出 resnet50
    python export_model.py --model resnet18      # 轻量、最快
    python export_model.py --model resnet50      # 精度更高（推荐）
    python export_model.py --model mobilenet_v3  # 移动端友好，体积小
    python export_model.py --list                # 查看所有可选模型
"""

from __future__ import annotations

import argparse
from pathlib import Path

OUT_DIR = Path("models")
IMAGE_SIZE = 224

# 可选模型：名称 -> (torchvision 构造函数名, 权重枚举名)
_MODELS = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "mobilenet_v3": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="导出分类模型到 OpenVINO IR")
    parser.add_argument("--model", default="resnet50", choices=list(_MODELS),
                        help="要导出的模型（默认 resnet50）")
    parser.add_argument("--list", action="store_true", help="列出可选模型后退出")
    args = parser.parse_args()

    if args.list:
        print("可选模型:")
        for name in _MODELS:
            print(f"  {name}")
        return

    try:
        import torch
        import torchvision
        import openvino as ov
    except ImportError as e:
        raise SystemExit(
            f"缺少依赖: {e.name}。请先安装: pip install torch torchvision openvino"
        )

    OUT_DIR.mkdir(exist_ok=True)
    ctor_name, weights_name = _MODELS[args.model]

    # 取 torchvision 里对应的构造函数与"最佳预训练权重"
    ctor = getattr(torchvision.models, ctor_name)
    weights_enum = getattr(torchvision.models, weights_name)
    weights = weights_enum.DEFAULT

    print(f"[export] 下载并加载 torchvision {ctor_name} (预训练权重)...")
    model = ctor(weights=weights)
    model.eval()

    # 固定 batch=1 的静态输入，便于 NPU 编译
    example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("[export] 转换为 OpenVINO 模型...")
    ov_model = ov.convert_model(model, example_input=example)

    xml_path = OUT_DIR / f"{args.model}.xml"
    ov.save_model(ov_model, xml_path)
    print(f"[export] 完成: {xml_path}")

    # 保存 ImageNet 类别标签（所有这些模型都用同一套 1000 类）
    labels = weights.meta["categories"]
    labels_path = OUT_DIR / "imagenet_classes.txt"
    labels_path.write_text("\n".join(labels), encoding="utf-8")
    print(f"[export] 标签文件: {labels_path}")
    print(f"[export] 该模型 Top-1 精度参考: {weights.meta.get('_metrics', {})}")


if __name__ == "__main__":
    main()
