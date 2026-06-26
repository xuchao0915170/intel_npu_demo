"""OpenVINO 推理引擎封装。

针对 Intel NPU 做了两点关键处理：
  1. 设备回退：NPU 不可用时按 NPU -> GPU -> CPU 顺序自动降级，方便在没有
     NPU 的开发机上也能跑通同一套代码。
  2. 静态 shape：Intel NPU 插件不支持动态 shape，编译前会把输入 reshape
     成固定尺寸。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import openvino as ov


# NPU 不可用时的降级顺序
_FALLBACK_ORDER = ["NPU", "GPU", "CPU"]


class NPUEngine:
    """封装模型加载、设备选择与推理的最小推理引擎。

    用法::

        engine = NPUEngine("model.xml", device="NPU", input_shape=(1, 3, 224, 224))
        outputs = engine.infer(input_array)
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "NPU",
        input_shape: Sequence[int] | None = None,
        config: dict | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.core = ov.Core()
        self.requested_device = device
        self.device = self._resolve_device(device)

        # 读取模型（支持 OpenVINO IR 的 .xml，也支持直接读 .onnx）
        self.model = self.core.read_model(self.model_path)

        # NPU 要求静态 shape：若指定了 input_shape 或模型本身是动态的则 reshape
        if input_shape is not None:
            self._reshape(input_shape)
        elif self.device == "NPU" and self._is_dynamic():
            raise ValueError(
                "NPU 不支持动态 shape，请通过 input_shape 指定固定输入尺寸，"
                "例如 input_shape=(1, 3, 224, 224)"
            )

        # 编译模型（编译失败时按降级顺序重试，例如 NPU 驱动不支持某算子）
        self.compiled_model = self._compile_with_fallback(config)
        self.input_port = self.compiled_model.inputs[0]
        self.output_ports = self.compiled_model.outputs

    def _compile_with_fallback(self, config: dict | None):
        """尝试在当前设备编译；失败则沿降级顺序换设备重试。

        NPU 存在但其驱动内置编译器不支持模型里的某个算子（如旧驱动的
        MaxPool）时，compile_model 会抛 RuntimeError —— 这里捕获后自动
        换到 GPU/CPU，保证开发机上整条链路仍能跑通。
        """
        available = self.core.available_devices
        # 候选设备：当前设备 + 其后的降级设备
        if self.device in _FALLBACK_ORDER:
            start = _FALLBACK_ORDER.index(self.device)
            candidates = [d for d in _FALLBACK_ORDER[start:] if d in available]
        else:
            candidates = [self.device]

        last_err: Exception | None = None
        for dev in candidates:
            # NPU 上默认用 FP16 精度提示，通常更快且占用更小
            compile_config = dict(config or {})
            if dev == "NPU":
                compile_config.setdefault("INFERENCE_PRECISION_HINT", "f16")
            try:
                compiled = self.core.compile_model(self.model, dev, compile_config)
                if dev != self.device:
                    print(f"[NPUEngine] 设备 '{self.device}' 编译失败，已回退到 '{dev}'")
                    self.device = dev
                return compiled
            except RuntimeError as e:
                last_err = e
                msg = str(e).splitlines()[-1] if str(e) else repr(e)
                print(f"[NPUEngine] 在 '{dev}' 上编译失败: {msg}")
        raise RuntimeError(
            f"在所有候选设备 {candidates} 上编译均失败"
        ) from last_err

    # ---------- 设备管理 ----------

    @property
    def available_devices(self) -> list[str]:
        return list(self.core.available_devices)

    def _resolve_device(self, device: str) -> str:
        """若请求的设备不可用，则沿降级顺序选择第一个可用设备。"""
        available = self.core.available_devices
        if device in available:
            return device

        print(f"[NPUEngine] 设备 '{device}' 不可用，当前可用设备: {available}")
        start = _FALLBACK_ORDER.index(device) + 1 if device in _FALLBACK_ORDER else 0
        for candidate in _FALLBACK_ORDER[start:]:
            if candidate in available:
                print(f"[NPUEngine] 自动回退到 '{candidate}'")
                return candidate
        # 实在没有就用 OpenVINO 报告的第一个设备
        if available:
            print(f"[NPUEngine] 回退到 '{available[0]}'")
            return available[0]
        raise RuntimeError("未发现任何 OpenVINO 可用设备")

    # ---------- shape 处理 ----------

    def _is_dynamic(self) -> bool:
        return any(inp.get_partial_shape().is_dynamic for inp in self.model.inputs)

    def _reshape(self, input_shape: Sequence[int]) -> None:
        first_input = self.model.inputs[0]
        self.model.reshape({first_input.get_any_name(): ov.PartialShape(list(input_shape))})

    # ---------- 推理 ----------

    def infer(self, data: np.ndarray) -> list[np.ndarray]:
        """单次同步推理，返回各输出端口的 numpy 数组列表。"""
        result = self.compiled_model({self.input_port: data})
        return [result[port] for port in self.output_ports]

    def benchmark(self, data: np.ndarray, runs: int = 50, warmup: int = 5) -> float:
        """简单测速，返回平均单次推理耗时（毫秒）。"""
        for _ in range(warmup):
            self.infer(data)
        start = time.perf_counter()
        for _ in range(runs):
            self.infer(data)
        elapsed = (time.perf_counter() - start) / runs * 1000
        return elapsed

    def __repr__(self) -> str:
        return (
            f"NPUEngine(model={self.model_path.name}, "
            f"device={self.device}, requested={self.requested_device})"
        )
