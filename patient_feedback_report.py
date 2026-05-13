import json
import sys

from core.progress_report import compare_progress_summaries
from core.patient_feedback import build_patient_feedback


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Kullanım: python patient_feedback_report.py eski.json yeni.json")
        sys.exit(1)

    old_summary = load_summary(sys.argv[1])
    new_summary = load_summary(sys.argv[2])

    report = compare_progress_summaries(old_summary, new_summary)
    feedback = build_patient_feedback(report, new_summary)

    print("=== TEKNIK RAPOR ===")
    print(report)
    print()
    print("=== HASTA GERI BILDIRIMI ===")
    print(feedback)