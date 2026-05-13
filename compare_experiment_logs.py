import json
import os
import sys
from typing import Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize(path: str) -> Dict[str, Any]:
    data = load_json(path)

    return {
        "file": os.path.basename(path),
        "patient_name": data.get("patient_name"),
        "exercise_code": data.get("exercise_code"),
        "model_name": data.get("model_name"),
        "target_angle": data.get("target_angle"),
        "frame_count": safe_get(data, "summary", "frame_count", default=0),
        "duration_sec": safe_get(data, "summary", "duration_sec", default=0),
        "completed_reps": safe_get(data, "summary", "completed_reps", default=0),
        "avg_angle": safe_get(data, "summary", "avg_angle", default=0),
        "max_angle": safe_get(data, "summary", "max_angle", default=0),
        "min_angle": safe_get(data, "summary", "min_angle", default=0),
        "std_angle": safe_get(data, "summary", "std_angle", default=0),
        "avg_fps": safe_get(data, "summary", "avg_fps", default=0),
        "min_fps": safe_get(data, "summary", "min_fps", default=0),
        "max_fps": safe_get(data, "summary", "max_fps", default=0),
        "completion_rate": safe_get(data, "summary", "completion_rate", default=0),
        "missing_angle_to_target": safe_get(data, "summary", "missing_angle_to_target", default=None),
        "pain_before": safe_get(data, "summary", "pain_before", default=None),
        "pain_after": safe_get(data, "summary", "pain_after", default=None),
    }


def print_summary(title: str, s: Dict[str, Any]):
    print("=" * 70)
    print(title)
    print("=" * 70)
    for k, v in s.items():
        print(f"{k:25}: {v}")
    print()


def compare(a: Dict[str, Any], b: Dict[str, Any]):
    print("=" * 70)
    print("KARŞILAŞTIRMA")
    print("=" * 70)

    numeric_fields = [
        "frame_count",
        "duration_sec",
        "completed_reps",
        "avg_angle",
        "max_angle",
        "min_angle",
        "std_angle",
        "avg_fps",
        "min_fps",
        "max_fps",
        "completion_rate",
    ]

    for field in numeric_fields:
        av = a.get(field)
        bv = b.get(field)

        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            diff = av - bv
            print(f"{field:25}: {a['model_name']}={av} | {b['model_name']}={bv} | fark={diff:.4f}")
        else:
            print(f"{field:25}: {a['model_name']}={av} | {b['model_name']}={bv}")

    print()
    print("YORUM")
    print("-" * 70)

    if a["avg_fps"] > b["avg_fps"]:
        print(f"- Daha hızlı model: {a['model_name']}")
    elif b["avg_fps"] > a["avg_fps"]:
        print(f"- Daha hızlı model: {b['model_name']}")
    else:
        print("- Ortalama FPS eşit.")

    if a["std_angle"] < b["std_angle"]:
        print(f"- Daha stabil açı ölçümü: {a['model_name']}")
    elif b["std_angle"] < a["std_angle"]:
        print(f"- Daha stabil açı ölçümü: {b['model_name']}")
    else:
        print("- Açısal stabilite eşit.")

    if a["max_angle"] > b["max_angle"]:
        print(f"- Daha yüksek maksimum açı ölçen: {a['model_name']}")
    elif b["max_angle"] > a["max_angle"]:
        print(f"- Daha yüksek maksimum açı ölçen: {b['model_name']}")
    else:
        print("- Maksimum açı eşit.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Kullanım:")
        print("python compare_experiment_logs.py mediapipe.json yolo.json")
        sys.exit(1)

    path1 = sys.argv[1]
    path2 = sys.argv[2]

    s1 = summarize(path1)
    s2 = summarize(path2)

    print_summary("LOG 1", s1)
    print_summary("LOG 2", s2)
    compare(s1, s2)