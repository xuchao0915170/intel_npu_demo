"""任务抽象基类。

一个"任务"负责把原始输入（如图片）转成模型输入张量（预处理），
以及把模型输出转成可读结果（后处理）。推理交给 NPUEngine。
新增任务（检测、分割等）只需继承本类实现两个方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .engine import NPUEngine


class BaseTask(ABC):
    def __init__(self, engine: NPUEngine) -> None:
        self.engine = engine

    @abstractmethod
    def preprocess(self, raw_input: Any) -> np.ndarray:
        """把原始输入转换为模型输入张量 (NCHW)。"""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, outputs: list[np.ndarray]) -> Any:
        """把模型原始输出转换为可读结果。"""
        raise NotImplementedError

    def run(self, raw_input: Any) -> Any:
        """端到端：预处理 -> 推理 -> 后处理。"""
        tensor = self.preprocess(raw_input)
        outputs = self.engine.infer(tensor)
        return self.postprocess(outputs)
