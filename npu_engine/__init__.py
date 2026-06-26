"""npu_engine 包：基于 OpenVINO 的推理引擎封装。"""

from .engine import NPUEngine
from .base_task import BaseTask

__all__ = ["NPUEngine", "BaseTask"]
