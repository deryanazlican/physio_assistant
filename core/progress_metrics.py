from typing import Any, Dict, Optional


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_symmetry_score(right_value: Optional[float], left_value: Optional[float]) -> Optional[float]:
    if right_value is None or left_value is None:
        return None

    denom = max(abs(right_value), abs(left_value), 1e-6)
    score = 1.0 - (abs(float(right_value) - float(left_value)) / denom)
    return round(clamp01(score), 4)


def build_progress_payload(
    *,
    exercise_code: str,
    reps: int = 0,
    target_reps: int = 10,
    done: bool = False,
    movement_name: str = "movement",
    movement_value: float = 0.0,
    movement_target: float = 0.0,
    movement_unit: str = "deg",
    quality_score: float = 0.0,
    max_movement_value: Optional[float] = None,
    avg_movement_value: Optional[float] = None,
    std_movement_value: Optional[float] = None,
    right_value: Optional[float] = None,
    left_value: Optional[float] = None,
    right_reps: Optional[int] = None,
    left_reps: Optional[int] = None,
    hold_time: Optional[float] = None,
    timer: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "exercise_code": exercise_code,
        "reps": int(reps),
        "target_reps": int(target_reps),
        "done": bool(done),
        "movement_name": movement_name,
        "movement_value": float(movement_value),
        "movement_target": float(movement_target),
        "movement_unit": movement_unit,
        "quality_score": round(float(quality_score), 4),
    }

    if max_movement_value is not None:
        payload["max_movement_value"] = float(max_movement_value)
    if avg_movement_value is not None:
        payload["avg_movement_value"] = float(avg_movement_value)
    if std_movement_value is not None:
        payload["std_movement_value"] = float(std_movement_value)

    if right_value is not None:
        payload["max_right_value"] = float(right_value)
    if left_value is not None:
        payload["max_left_value"] = float(left_value)

    if right_reps is not None:
        payload["right_reps"] = int(right_reps)
    if left_reps is not None:
        payload["left_reps"] = int(left_reps)

    if right_value is not None and left_value is not None:
        payload["symmetry_score"] = compute_symmetry_score(right_value, left_value)

    if hold_time is not None:
        payload["hold_time"] = float(hold_time)
    if timer is not None:
        payload["timer"] = float(timer)

    if extra:
        payload.update(extra)

    return payload