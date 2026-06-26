"""图像分类任务示例（ImageNet 1000 类，ResNet 风格预处理）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from npu_engine import BaseTask

# ImageNet 标准归一化参数
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ClassificationTask(BaseTask):
    def __init__(self, engine, labels: list[str] | None = None, image_size: int = 224):
        super().__init__(engine)
        self.image_size = image_size
        self.labels = labels

    def preprocess(self, image_path: str | Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD          # HWC 归一化
        arr = arr.transpose(2, 0, 1)        # HWC -> CHW
        return arr[np.newaxis, ...].astype(np.float32)  # 加 batch 维 -> NCHW

    def postprocess(self, outputs: list[np.ndarray], top_k: int = 5):
        logits = outputs[0].reshape(-1)
        probs = _softmax(logits)
        top_idx = probs.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            label = self.labels[idx] if self.labels and idx < len(self.labels) else str(idx)
            results.append((int(idx), label, float(probs[idx])))
        return results


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
