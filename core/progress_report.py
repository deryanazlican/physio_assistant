from typing import Any, Dict, Optional


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compare_progress_summaries(old_summary: Dict[str, Any], new_summary: Dict[str, Any]) -> Dict[str, Any]:
    old_max = _safe_num(old_summary.get("max_movement_value", old_summary.get("movement_value", 0.0)))
    new_max = _safe_num(new_summary.get("max_movement_value", new_summary.get("movement_value", 0.0)))

    old_avg = _safe_num(old_summary.get("avg_movement_value", old_summary.get("movement_value", 0.0)))
    new_avg = _safe_num(new_summary.get("avg_movement_value", new_summary.get("movement_value", 0.0)))

    old_reps = int(old_summary.get("completed_reps_total", old_summary.get("completed_reps", 0)) or 0)
    new_reps = int(new_summary.get("completed_reps_total", new_summary.get("completed_reps", 0)) or 0)

    old_quality = _safe_num(old_summary.get("quality_score", 0.0))
    new_quality = _safe_num(new_summary.get("quality_score", 0.0))

    old_symmetry = old_summary.get("symmetry_score")
    new_symmetry = new_summary.get("symmetry_score")

    old_pain = old_summary.get("pain_after")
    new_pain = new_summary.get("pain_after")

    report = {
        "movement_change": round(new_max - old_max, 2),
        "average_movement_change": round(new_avg - old_avg, 2),
        "reps_change": int(new_reps - old_reps),
        "quality_change": round(new_quality - old_quality, 4),
        "symmetry_change": None,
        "pain_change": None,
        "old_max_movement": round(old_max, 2),
        "new_max_movement": round(new_max, 2),
        "old_reps": old_reps,
        "new_reps": new_reps,
    }

    if old_symmetry is not None and new_symmetry is not None:
        report["symmetry_change"] = round(_safe_num(new_symmetry) - _safe_num(old_symmetry), 4)

    if old_pain is not None and new_pain is not None:
        report["pain_change"] = round(_safe_num(new_pain) - _safe_num(old_pain), 2)

    movement_change = report["movement_change"]
    reps_change = report["reps_change"]
    pain_change = report["pain_change"]

    notes = []

    if movement_change > 0:
        notes.append("Movement range improved.")
    elif movement_change < 0:
        notes.append("Movement range decreased.")
    else:
        notes.append("Movement range unchanged.")

    if reps_change > 0:
        notes.append("Completed repetitions increased.")
    elif reps_change < 0:
        notes.append("Completed repetitions decreased.")

    if pain_change is not None:
        if pain_change < 0:
            notes.append("Pain decreased after exercise.")
        elif pain_change > 0:
            notes.append("Pain increased after exercise.")

    if report["symmetry_change"] is not None:
        if report["symmetry_change"] > 0:
            notes.append("Right-left symmetry improved.")
        elif report["symmetry_change"] < 0:
            notes.append("Right-left asymmetry increased.")

    report["notes"] = notes
    return report