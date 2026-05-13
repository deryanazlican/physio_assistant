from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "experiment_logs")
ARCHIVE_DIR = os.path.join(LOGS_DIR, "archived_old_logs")


def slugify_patient_name(name: str) -> str:
    name = (name or "UNKNOWN").strip().upper()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Z0-9_ÇĞİÖŞÜ]", "", name)
    return name or "UNKNOWN"


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception as e:
        print(f"[WARN] JSON okunamadı: {path} -> {e}")
        return None


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_new_patient_format(data: Dict[str, Any]) -> bool:
    return isinstance(data.get("sessions"), list)


def looks_like_old_single_session(data: Dict[str, Any]) -> bool:
    return (
        "patient_name" in data
        and "exercise_code" in data
        and "summary" in data
        and "frames" in data
    )


def build_session_from_old_log(data: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    created_at = data.get("created_at")
    if not created_at:
        try:
            created_at = datetime.fromtimestamp(os.path.getmtime(source_path)).isoformat()
        except Exception:
            created_at = datetime.now().isoformat()

    session_id = data.get("session_id")
    if not session_id:
        exercise_code = str(data.get("exercise_code", "UNKNOWN"))
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exercise_code}"

    return {
        "session_id": session_id,
        "patient_name": data.get("patient_name", "UNKNOWN"),
        "exercise_code": data.get("exercise_code", "UNKNOWN"),
        "model_name": data.get("model_name", "unknown"),
        "target_angle": data.get("target_angle"),
        "target_reps": data.get("target_reps", 10),
        "start_timestamp": data.get("start_timestamp"),
        "created_at": created_at,
        "summary": data.get("summary", {}) or {},
        "frames": data.get("frames", []) or [],
    }


def make_session_fingerprint(session: Dict[str, Any]) -> str:
    patient_name = str(session.get("patient_name", "")).strip().upper()
    exercise_code = str(session.get("exercise_code", "")).strip().upper()
    created_at = str(session.get("created_at", "")).strip()
    start_ts = str(session.get("start_timestamp", "")).strip()

    summary = session.get("summary", {}) or {}
    completed_reps = str(summary.get("completed_reps", ""))
    duration_sec = str(summary.get("duration_sec", ""))
    pain_before = str(summary.get("pain_before", ""))
    pain_after = str(summary.get("pain_after", ""))

    return "|".join([
        patient_name,
        exercise_code,
        created_at,
        start_ts,
        completed_reps,
        duration_sec,
        pain_before,
        pain_after,
    ])


def load_or_create_patient_file(patient_name: str) -> tuple[str, Dict[str, Any]]:
    safe_name = slugify_patient_name(patient_name)
    patient_file = os.path.join(LOGS_DIR, f"{safe_name}.json")

    if os.path.exists(patient_file):
        data = load_json(patient_file)
        if data and is_new_patient_format(data):
            data.setdefault("patient_name", patient_name)
            data.setdefault("created_at", datetime.now().isoformat())
            data.setdefault("updated_at", datetime.now().isoformat())
            data.setdefault("sessions", [])
            return patient_file, data

    data = {
        "patient_name": patient_name,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "sessions": [],
    }
    return patient_file, data


def merge_old_logs(archive_old_files: bool = True) -> None:
    if not os.path.isdir(LOGS_DIR):
        print(f"[WARN] Klasör yok: {LOGS_DIR}")
        return

    if archive_old_files:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)

    all_files = [
        os.path.join(LOGS_DIR, f)
        for f in os.listdir(LOGS_DIR)
        if f.lower().endswith(".json")
    ]

    if not all_files:
        print("[INFO] Birleştirilecek JSON bulunamadı.")
        return

    patient_buckets: Dict[str, List[Dict[str, Any]]] = {}
    old_files_to_archive: List[str] = []
    skipped_new_format = 0
    skipped_unknown = 0

    for path in all_files:
        filename = os.path.basename(path)

        if filename == os.path.basename(ARCHIVE_DIR):
            continue

        data = load_json(path)
        if not data:
            continue

        if is_new_patient_format(data):
            skipped_new_format += 1
            continue

        if not looks_like_old_single_session(data):
            skipped_unknown += 1
            print(f"[WARN] Tanınmayan format, atlandı: {filename}")
            continue

        patient_name = str(data.get("patient_name", "UNKNOWN")).strip() or "UNKNOWN"
        session = build_session_from_old_log(data, path)

        patient_buckets.setdefault(patient_name, []).append(session)
        old_files_to_archive.append(path)

    total_added = 0

    for patient_name, incoming_sessions in patient_buckets.items():
        patient_file, patient_data = load_or_create_patient_file(patient_name)
        existing_sessions = patient_data.get("sessions", [])

        existing_fingerprints = {
            make_session_fingerprint(sess) for sess in existing_sessions
        }

        new_sessions = []
        for sess in incoming_sessions:
            fp = make_session_fingerprint(sess)
            if fp not in existing_fingerprints:
                new_sessions.append(sess)
                existing_fingerprints.add(fp)

        combined_sessions = existing_sessions + new_sessions
        combined_sessions.sort(key=lambda s: str(s.get("created_at", "")))

        patient_data["sessions"] = combined_sessions
        patient_data["updated_at"] = datetime.now().isoformat()

        save_json(patient_file, patient_data)

        total_added += len(new_sessions)
        print(
            f"[OK] {patient_name} -> {os.path.basename(patient_file)} | "
            f"eklenen: {len(new_sessions)} | toplam: {len(combined_sessions)}"
        )

    if archive_old_files:
        for old_path in old_files_to_archive:
            try:
                target_path = os.path.join(ARCHIVE_DIR, os.path.basename(old_path))
                if os.path.abspath(old_path) == os.path.abspath(target_path):
                    continue
                shutil.move(old_path, target_path)
            except Exception as e:
                print(f"[WARN] Arşive taşınamadı: {old_path} -> {e}")

    print("\n=== ÖZET ===")
    print(f"Yeni format olduğu için atlanan hasta dosyası: {skipped_new_format}")
    print(f"Tanınmayan format olduğu için atlanan dosya: {skipped_unknown}")
    print(f"Toplam eklenen session sayısı: {total_added}")
    if archive_old_files:
        print(f"Eski dosyalar taşındı: {ARCHIVE_DIR}")


if __name__ == "__main__":
    merge_old_logs(archive_old_files=True)