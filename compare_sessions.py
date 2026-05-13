# compare_sessions.py
import json
import sys
from core.progress_report import compare_progress_summaries

def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]

old_path = sys.argv[1]
new_path = sys.argv[2]

old_summary = load_summary(old_path)
new_summary = load_summary(new_path)

report = compare_progress_summaries(old_summary, new_summary)
print(report)