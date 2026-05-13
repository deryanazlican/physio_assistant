from typing import Dict, List, Optional
import math


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_val = safe_mean(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def summarize_session(frame_logs: List[Dict]) -> Dict:
    angles = [x["angle"] for x in frame_logs if x["angle"] is not None]
    fps_values = [x["fps"] for x in frame_logs if x["fps"] is not None]
    complete_flags = [1 for x in frame_logs if x["is_complete"]]

    return {
        "frame_count": len(frame_logs),
        "avg_angle": round(safe_mean(angles), 2),
        "max_angle": round(max(angles), 2) if angles else 0.0,
        "min_angle": round(min(angles), 2) if angles else 0.0,
        "std_angle": round(safe_std(angles), 2),
        "avg_fps": round(safe_mean(fps_values), 2),
        "completion_count": len(complete_flags),
        "completion_rate": round(len(complete_flags) / len(frame_logs), 4) if frame_logs else 0.0
    }