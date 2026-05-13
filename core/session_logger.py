import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class SessionLogger:
    def __init__(self, user_name: str, exercise_name: str, model_name: str, log_dir: str = "logs"):
        self.user_name = user_name
        self.exercise_name = exercise_name
        self.model_name = model_name
        self.log_dir = log_dir
        self.frames: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}

        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(
            self.log_dir,
            f"{self.user_name}_{self.exercise_name}_{self.model_name}_{timestamp}.json"
        )

    def log_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        angle: Optional[float],
        target_angle: Optional[float],
        is_complete: bool,
        rep_count: int,
        fps: float,
        confidence: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        item = {
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 3),
            "angle": None if angle is None else round(angle, 2),
            "target_angle": target_angle,
            "is_complete": is_complete,
            "rep_count": rep_count,
            "fps": round(fps, 2),
            "confidence": confidence,
            "extra": extra or {}
        }
        self.frames.append(item)

    def set_summary(self, summary: Dict[str, Any]) -> None:
        self.summary = summary

    def save(self) -> str:
        payload = {
            "user_name": self.user_name,
            "exercise_name": self.exercise_name,
            "model_name": self.model_name,
            "created_at": datetime.now().isoformat(),
            "summary": self.summary,
            "frames": self.frames
        }

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return self.file_path