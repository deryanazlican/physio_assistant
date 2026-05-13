import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt


def filter_patient_records(history: List[Dict[str, Any]], patient_name: str) -> List[Dict[str, Any]]:
    return [item for item in history if item.get("patient_name") == patient_name]


def build_progress_lists(records: List[Dict[str, Any]]):
    x = list(range(1, len(records) + 1))
    pain_before = []
    pain_after = []
    quality = []

    for item in records:
        data = item.get("data", {}) or {}
        pain_before.append(float(data.get("current_pain", 0) or 0))
        pain_after.append(float(data.get("pain_after", 0) or 0))
        quality.append(float(data.get("quality", 0) or 0))

    return x, pain_before, pain_after, quality


def generate_progress_dashboard(history: List[Dict[str, Any]], patient_name: str, output_dir: str) -> str | None:
    records = filter_patient_records(history, patient_name)

    if not records:
        print("Dashboard için kayıt bulunamadı.")
        return None

    x, pain_before, pain_after, quality = build_progress_lists(records)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{patient_name}_progress_dashboard.png")

    plt.figure(figsize=(10, 6))
    plt.plot(x, pain_before, marker="o", label="Pain Before")
    plt.plot(x, pain_after, marker="o", label="Pain After")
    plt.plot(x, quality, marker="o", label="Quality")
    plt.xlabel("Session")
    plt.ylabel("Value")
    plt.title(f"Progress Dashboard - {patient_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path