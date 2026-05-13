from __future__ import annotations

import json
from pathlib import Path


SKIP_FILES = {
    "llm_eval_results.json",
    "llm_eval_summary.csv",
}


def load_experiment_logs(log_dir: str = "experiment_logs") -> list[dict]:
    root = Path(log_dir)
    results = []

    if not root.exists():
        return results

    for file in root.glob("*.json"):
        if file.name in SKIP_FILES:
            continue

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # sadece gerçek session log dict'lerini kabul et
            if not isinstance(data, dict):
                print(f"Atlandı (dict değil): {file.name}")
                continue

            # summary alanı yoksa büyük ihtimalle session log değildir
            if "summary" not in data:
                print(f"Atlandı (summary yok): {file.name}")
                continue

            results.append({
                "file_name": file.name,
                "file_path": str(file),
                "data": data,
            })

        except Exception as e:
            results.append({
                "file_name": file.name,
                "file_path": str(file),
                "error": str(e),
            })

    return results