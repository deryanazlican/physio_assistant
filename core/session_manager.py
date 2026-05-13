import json
import re
from datetime import datetime
from pathlib import Path


class SessionManager:
    def __init__(self, data_folder="data/sessions"):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, name: str) -> str:
        name = (name or "UNKNOWN").strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
        return name or "UNKNOWN"

    def start_session(self, patient_name: str, complaint: str, condition: str, plan: dict | None = None) -> dict:
        return {
            "patient_name": patient_name,
            "complaint": complaint,
            "condition": condition,
            "created_at": datetime.now().isoformat(),
            "ended_at": None,
            "plan": plan or {},
            "exercise_results": [],
            "notes": [],
        }

    def add_exercise_result(
        self,
        session: dict,
        exercise_code: str,
        target_reps: int = 0,
        completed_reps: int = 0,
        duration_sec: int = 0,
        status: str = "done",
    ):
        session.setdefault("exercise_results", []).append({
            "exercise_code": exercise_code,
            "target_reps": int(target_reps),
            "completed_reps": int(completed_reps),
            "duration_sec": int(duration_sec),
            "status": status,
            "timestamp": datetime.now().isoformat(),
        })

    def add_note(self, session: dict, note: str):
        if note:
            session.setdefault("notes", []).append(note)

    def update_plan(self, session: dict, plan: dict):
        session["plan"] = plan

    def end_session(self, session: dict):
        session["ended_at"] = datetime.now().isoformat()

    def save_session(self, session: dict) -> str:
        patient = self._safe_name(session.get("patient_name", "UNKNOWN"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.data_folder / f"{patient}_session_{timestamp}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

        return str(filepath)

    def save_active_session(self, session: dict) -> str:
        patient = self._safe_name(session.get("patient_name", "UNKNOWN"))
        filepath = self.data_folder / f"{patient}_active_session.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

        return str(filepath)

    def load_active_session(self, patient_name: str):
        patient = self._safe_name(patient_name)
        filepath = self.data_folder / f"{patient}_active_session.json"

        if not filepath.exists():
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def finalize_session(self, session: dict) -> str:
        self.end_session(session)
        final_path = self.save_session(session)

        patient = self._safe_name(session.get("patient_name", "UNKNOWN"))
        active_path = self.data_folder / f"{patient}_active_session.json"
        if active_path.exists():
            active_path.unlink()

        return final_path