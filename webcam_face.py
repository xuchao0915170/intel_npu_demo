"""摄像头实时人脸检测（Intel NPU + OpenVINO）。

打开默认摄像头，逐帧在 NPU 上做人脸检测，把检测框和实时 FPS 画到画面上。
按 q 或 ESC 退出。

用法:
    python webcam_face.py
    python webcam_face.py --device GPU --conf 0.6 --camera 1
"""

from __future__ import annotations

import argparse
import time

import cv2

from npu_engine import NPUEngine
from tasks.face_detection import FaceDetectionTask

MODEL = "models/face-detection-0200.xml"
INPUT_SIZE = 256


def main() -> None:
    parser = argparse.ArgumentParser(description="摄像头实时人脸检测")
    parser.add_argument("--device", default="NPU", help="NPU/GPU/CPU")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号（默认 0）")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--model", default=MODEL, help="人脸检测模型路径")
    args = parser.parse_args()

    # 初始化推理引擎与任务
    engine = NPUEngine(args.model, device=args.device,
                       input_shape=(1, 3, INPUT_SIZE, INPUT_SIZE))
    print(f"[webcam] {engine}")
    task = FaceDetectionTask(engine, input_size=INPUT_SIZE, conf_threshold=args.conf)

    # 打开摄像头（Windows 上用 CAP_DSHOW 避免启动慢）
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头 {args.camera}，换个 --camera 编号试试")

    print("[webcam] 开始检测，按 q 或 ESC 退出")
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[webcam] 读取帧失败，退出")
            break

        t0 = time.perf_counter()
        faces = task.run_frame(frame)          # NPU 推理
        infer_ms = (time.perf_counter() - t0) * 1000
        fps = 0.9 * fps + 0.1 * (1000.0 / max(infer_ms, 1e-3))  # 平滑 FPS

        # 画框 + 置信度
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{f['confidence']:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 顶部状态栏
        cv2.putText(frame,
                    f"{engine.device}  faces={len(faces)}  {infer_ms:.1f}ms  {fps:.0f}FPS",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("NPU Face Detection (press q/ESC to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q 或 ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[webcam] 已退出")


if __name__ == "__main__":
    main()
