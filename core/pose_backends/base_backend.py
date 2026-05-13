from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BasePoseBackend(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def process(self, frame) -> Dict[str, Any]:
        """
        Çıktı formatı:
        {
            "landmarks": ...,
            "pose_detected": bool,
            "confidence": float | None,
            "raw_result": any
        }
        """
        pass

    @abstractmethod
    def draw(self, frame, result: Dict[str, Any]) -> None:
        pass