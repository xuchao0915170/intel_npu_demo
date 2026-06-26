"""人脸检测任务（OpenVINO Open Model Zoo: face-detection-0200, SSD 结构）。

只做"画面里有没有脸、在哪个位置"，不做身份识别。
模型输入: 1x3x256x256 (BGR, 0-255, 无归一化)
模型输出: [1,1,N,7]，每行 = [image_id, label, conf, xmin, ymin, xmax, ymax]
          坐标为归一化到 [0,1] 的比例值。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from npu_engine import BaseTask


class FaceDetectionTask(BaseTask):
    def __init__(self, engine, input_size: int = 256, conf_threshold: float = 0.5):
        super().__init__(engine)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self._orig_size: tuple[int, int] = (0, 0)  # (W, H) 原图尺寸，画框时用

    def preprocess(self, image_path: str | Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        self._orig_size = img.size  # (W, H)
        resized = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)   # HWC, RGB, 0-255
        arr = arr[:, :, ::-1]                          # RGB -> BGR（该模型按 BGR 训练）
        arr = arr.transpose(2, 0, 1)                   # HWC -> CHW
        return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """直接处理 OpenCV 摄像头帧（已是 BGR 的 HWC ndarray）。

        摄像头实时场景每帧都调用，避免走文件读写。
        """
        import cv2
        h, w = frame_bgr.shape[:2]
        self._orig_size = (w, h)
        resized = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        arr = resized.astype(np.float32).transpose(2, 0, 1)  # 已是 BGR，直接 HWC->CHW
        return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)

    def run_frame(self, frame_bgr: np.ndarray) -> list[dict]:
        """端到端处理一帧摄像头画面，返回人脸列表。"""
        tensor = self.preprocess_frame(frame_bgr)
        outputs = self.engine.infer(tensor)
        return self.postprocess(outputs)

    def postprocess(self, outputs: list[np.ndarray]) -> list[dict]:
        det = outputs[0].reshape(-1, 7)  # [N, 7]
        w, h = self._orig_size
        faces = []
        for row in det:
            _, _, conf, x_min, y_min, x_max, y_max = row
            if conf < self.conf_threshold:
                continue
            faces.append({
                "confidence": float(conf),
                # 归一化坐标 -> 原图像素坐标
                "box": (
                    int(x_min * w), int(y_min * h),
                    int(x_max * w), int(y_max * h),
                ),
            })
        return faces

    def draw(self, image_path: str | Path, faces: list[dict],
             out_path: str | Path = "faces_out.jpg") -> Path:
        """把检测框画到原图并保存。"""
        img = Image.open(image_path).convert("RGB")
        drawer = ImageDraw.Draw(img)
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            drawer.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            drawer.text((x1, max(0, y1 - 12)), f"{f['confidence']:.2f}", fill=(0, 255, 0))
        out = Path(out_path)
        img.save(out)
        return out
