from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


def _slugify_patient_name(name: str) -> str:
    name = (name or "UNKNOWN").strip().upper()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Z0-9_ÇĞİÖŞÜ]", "", name)
    return name or "UNKNOWN"


class ExperimentLogger:
    def __init__(self, logs_dir: str):
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

        self.current_session: Optional[Dict[str, Any]] = None
        self.current_patient_name: Optional[str] = None
        self.current_patient_file: Optional[str] = None

    def _get_patient_file_path(self, patient_name: str) -> str:
        safe_name = _slugify_patient_name(patient_name)
        return os.path.join(self.logs_dir, f"{safe_name}.json")

    def _load_patient_log(self, patient_name: str) -> Dict[str, Any]:
        path = self._get_patient_file_path(patient_name)

        if not os.path.exists(path):
            return {
                "patient_name": patient_name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "sessions": []
            }

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Patient log root dict değil")

            if "sessions" not in data or not isinstance(data["sessions"], list):
                data["sessions"] = []

            data.setdefault("patient_name", patient_name)
            data.setdefault("created_at", datetime.now().isoformat())
            data["updated_at"] = datetime.now().isoformat()

            return data

        except Exception as e:
            print(f"[ExperimentLogger] Patient log okunamadı, yeni yapı oluşturuluyor: {e}")
            return {
                "patient_name": patient_name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "sessions": []
            }

    def _save_patient_log(self, patient_name: str, data: Dict[str, Any]) -> str:
        path = self._get_patient_file_path(patient_name)
        data["updated_at"] = datetime.now().isoformat()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return path

    def start_session(
        self,
        patient_name: str,
        exercise_code: str,
        model_name: str,
        target_angle: Optional[float] = None,
        target_reps: int = 10
    ) -> None:
        now = datetime.now().isoformat()
        ts = time.time()

        self.current_patient_name = patient_name or "UNKNOWN"
        self.current_patient_file = self._get_patient_file_path(self.current_patient_name)

        self.current_session = {
            "session_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exercise_code}",
            "patient_name": self.current_patient_name,
            "exercise_code": exercise_code,
            "model_name": model_name,
            "target_angle": target_angle,
            "target_reps": target_reps,
            "start_timestamp": ts,
            "created_at": now,
            "summary": {},
            "frames": []
        }

    def log_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        fps: Optional[float],
        angle: Optional[float],
        reps: int,
        is_complete: bool,
        confidence: Optional[float],
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.current_session is None:
            return

        self.current_session["frames"].append({
            "frame_index": frame_index,
            "timestamp_sec": round(float(timestamp_sec), 3),
            "fps": round(float(fps), 2) if fps is not None else None,
            "angle": round(float(angle), 2) if angle is not None else None,
            "reps": int(reps),
            "is_complete": bool(is_complete),
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "extra": extra or {}
        })

    def finish_session(
        self,
        completed_reps: int,
        duration_sec: float,
        pain_before: Optional[int],
        pain_after: Optional[int],
        extra_summary: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        if self.current_session is None or self.current_patient_name is None:
            return None

        frames = self.current_session.get("frames", [])

        valid_angles = [
            float(fr["angle"])
            for fr in frames
            if fr.get("angle") is not None
        ]

        fps_values = [
            float(fr["fps"])
            for fr in frames
            if fr.get("fps") is not None
        ]

        completion_frame_count = sum(1 for fr in frames if fr.get("is_complete") is True)
        frame_count = len(frames)

        summary = {
            "frame_count": frame_count,
            "completed_reps": int(completed_reps),
            "duration_sec": round(float(duration_sec), 2),
            "avg_angle": round(sum(valid_angles) / len(valid_angles), 2) if valid_angles else 0.0,
            "max_angle": round(max(valid_angles), 2) if valid_angles else 0.0,
            "min_angle": round(min(valid_angles), 2) if valid_angles else 0.0,
            "std_angle": round(self._std(valid_angles), 2) if valid_angles else 0.0,
            "avg_fps": round(sum(fps_values) / len(fps_values), 2) if fps_values else 0.0,
            "min_fps": round(min(fps_values), 2) if fps_values else 0.0,
            "max_fps": round(max(fps_values), 2) if fps_values else 0.0,
            "completion_frame_count": completion_frame_count,
            "completion_rate": round(completion_frame_count / frame_count, 4) if frame_count > 0 else 0.0,
            "pain_before": pain_before,
            "pain_after": pain_after,
        }

        if extra_summary:
            summary.update(extra_summary)

        self.current_session["summary"] = summary

        patient_log = self._load_patient_log(self.current_patient_name)
        patient_log["sessions"].append(self.current_session)

        saved_path = self._save_patient_log(self.current_patient_name, patient_log)

        self.current_session = None
        self.current_patient_name = None
        self.current_patient_file = None

        return saved_path

    def update_last_session_pain_after(self, patient_name: str, pain_after: int) -> Optional[str]:
        patient_log = self._load_patient_log(patient_name)
        sessions = patient_log.get("sessions", [])

        if not sessions:
            return None

        last_session = sessions[-1]
        if "summary" not in last_session or not isinstance(last_session["summary"], dict):
            last_session["summary"] = {}

        last_session["summary"]["pain_after"] = int(pain_after)

        saved_path = self._save_patient_log(patient_name, patient_log)
        return saved_path

    @staticmethod
    def _std(values: List[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return var ** 0.5