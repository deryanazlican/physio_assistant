from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "experiment_logs"   # kendi proje klasörüne göre değiştir
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "pain_predictor_ml.joblib"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] JSON okunamadı: {path} -> {e}")
        return None


def find_log_files(logs_dir: Path) -> List[Path]:
    if not logs_dir.exists():
        print(f"[WARN] Log klasörü bulunamadı: {logs_dir}")
        return []
    return sorted(logs_dir.rglob("*.json"))


def extract_frame_features(frames: List[Dict[str, Any]]) -> Dict[str, float]:
    if not frames:
        return {
            "valid_angle_ratio": 0.0,
            "mean_confidence": 0.0,
            "low_confidence_ratio": 0.0,
            "angle_range_frames": 0.0,
            "angle_mean_frames": 0.0,
            "angle_std_frames": 0.0,
            "early_angle_mean": 0.0,
            "late_angle_mean": 0.0,
            "angle_trend_delta": 0.0,
            "rep_speed": 0.0,
            "complete_frame_ratio": 0.0,
        }

    angles: List[float] = []
    confidences: List[float] = []
    complete_flags: List[int] = []
    rep_values: List[int] = []
    timestamps: List[float] = []

    valid_angle_count = 0
    low_conf_count = 0

    for fr in frames:
        angle = fr.get("angle")
        conf = safe_float(fr.get("confidence", 0.0), 0.0)
        is_complete = bool(fr.get("is_complete", False))
        reps = safe_int(fr.get("reps", 0), 0)
        ts = safe_float(fr.get("timestamp_sec", 0.0), 0.0)

        confidences.append(conf)
        complete_flags.append(1 if is_complete else 0)
        rep_values.append(reps)
        timestamps.append(ts)

        if conf < 0.5:
            low_conf_count += 1

        if angle is not None:
            try:
                angle_f = float(angle)
                angles.append(angle_f)
                valid_angle_count += 1
            except Exception:
                pass

    total_frames = max(1, len(frames))
    valid_angle_ratio = valid_angle_count / total_frames
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    low_conf_ratio = low_conf_count / total_frames
    complete_frame_ratio = float(np.mean(complete_flags)) if complete_flags else 0.0

    if angles:
        angle_mean = float(np.mean(angles))
        angle_std = float(np.std(angles))
        angle_range = float(np.max(angles) - np.min(angles))

        third = max(1, len(angles) // 3)
        early_mean = float(np.mean(angles[:third]))
        late_mean = float(np.mean(angles[-third:]))
        angle_trend_delta = late_mean - early_mean
    else:
        angle_mean = 0.0
        angle_std = 0.0
        angle_range = 0.0
        early_mean = 0.0
        late_mean = 0.0
        angle_trend_delta = 0.0

    max_reps = max(rep_values) if rep_values else 0
    duration_sec = max(timestamps) if timestamps else 0.0
    rep_speed = (max_reps / duration_sec) if duration_sec > 0 else 0.0

    return {
        "valid_angle_ratio": round(valid_angle_ratio, 4),
        "mean_confidence": round(mean_confidence, 4),
        "low_confidence_ratio": round(low_conf_ratio, 4),
        "angle_range_frames": round(angle_range, 4),
        "angle_mean_frames": round(angle_mean, 4),
        "angle_std_frames": round(angle_std, 4),
        "early_angle_mean": round(early_mean, 4),
        "late_angle_mean": round(late_mean, 4),
        "angle_trend_delta": round(angle_trend_delta, 4),
        "rep_speed": round(rep_speed, 4),
        "complete_frame_ratio": round(complete_frame_ratio, 4),
    }


def build_row_from_log(log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary = log_data.get("summary", {}) or {}
    frames = log_data.get("frames", []) or []

    pain_before = summary.get("pain_before", None)
    pain_after = summary.get("pain_after", None)

    if pain_before is None or pain_after is None:
        return None

    frame_features = extract_frame_features(frames)

    row = {
        "patient_name": str(log_data.get("patient_name", "UNKNOWN")),
        "exercise_code": str(log_data.get("exercise_code", "UNKNOWN")),
        "model_name": str(log_data.get("model_name", "unknown")),
        "target_angle": safe_float(log_data.get("target_angle", 0.0)),
        "target_reps": safe_int(log_data.get("target_reps", 0)),

        "frame_count": safe_int(summary.get("frame_count", 0)),
        "completed_reps": safe_int(summary.get("completed_reps", 0)),
        "duration_sec": safe_float(summary.get("duration_sec", 0.0)),
        "avg_angle": safe_float(summary.get("avg_angle", 0.0)),
        "max_angle": safe_float(summary.get("max_angle", 0.0)),
        "min_angle": safe_float(summary.get("min_angle", 0.0)),
        "std_angle": safe_float(summary.get("std_angle", 0.0)),
        "avg_fps": safe_float(summary.get("avg_fps", 0.0)),
        "min_fps": safe_float(summary.get("min_fps", 0.0)),
        "max_fps": safe_float(summary.get("max_fps", 0.0)),
        "completion_frame_count": safe_int(summary.get("completion_frame_count", 0)),
        "completion_rate": safe_float(summary.get("completion_rate", 0.0)),
        "pain_before": safe_float(pain_before, 0.0),
        "missing_angle_to_target": safe_float(summary.get("missing_angle_to_target", 0.0)),
        "movement_name": str(summary.get("movement_name", "unknown")),
        "movement_value": safe_float(summary.get("movement_value", 0.0)),
        "movement_target": safe_float(summary.get("movement_target", 0.0)),
        "quality_score": safe_float(summary.get("quality_score", 0.0)),
        "max_movement_value": safe_float(summary.get("max_movement_value", 0.0)),
        "avg_movement_value": safe_float(summary.get("avg_movement_value", 0.0)),
        "completed_reps_total": safe_int(summary.get("completed_reps_total", 0)),
        "max_right_value": safe_float(summary.get("max_right_value", 0.0)),
        "max_left_value": safe_float(summary.get("max_left_value", 0.0)),
        "right_reps": safe_int(summary.get("right_reps", 0)),
        "left_reps": safe_int(summary.get("left_reps", 0)),
        "symmetry_score": safe_float(summary.get("symmetry_score", 0.0)),

        "pain_after": safe_float(pain_after, 0.0),
    }

    row.update(frame_features)

    row["pain_delta"] = row["pain_after"] - row["pain_before"]
    row["target_hit_ratio"] = (row["max_angle"] / row["target_angle"]) if row["target_angle"] > 0 else 0.0
    row["rep_completion_ratio"] = (row["completed_reps"] / row["target_reps"]) if row["target_reps"] > 0 else 0.0
    row["right_left_gap"] = abs(row["max_right_value"] - row["max_left_value"])

    return row


def build_dataframe_from_logs(log_files: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    skipped = 0

    for path in log_files:
        data = load_json_file(path)
        if not data:
            continue

        row = build_row_from_log(data)
        if row is None:
            skipped += 1
            continue

        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"[INFO] Toplam log dosyası: {len(log_files)}")
    print(f"[INFO] Eğitime alınan kayıt: {len(df)}")
    print(f"[INFO] pain_after / pain_before eksik olduğu için atlanan kayıt: {skipped}")

    return df


def create_pipeline(categorical_features: List[str], numeric_features: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train(logs_dir: Path = LOGS_DIR) -> None:
    log_files = find_log_files(logs_dir)
    df = build_dataframe_from_logs(log_files)

    if len(df) < 20:
        print("[WARN] Model eğitmek için veri az. En az 20-30 etiketli kayıt önerilir.")
        if len(df) > 0:
            print(df.head())
        return

    feature_cols = [
        "patient_name",
        "exercise_code",
        "model_name",
        "target_angle",
        "target_reps",
        "frame_count",
        "completed_reps",
        "duration_sec",
        "avg_angle",
        "max_angle",
        "min_angle",
        "std_angle",
        "avg_fps",
        "min_fps",
        "max_fps",
        "completion_frame_count",
        "completion_rate",
        "pain_before",
        "missing_angle_to_target",
        "movement_name",
        "movement_value",
        "movement_target",
        "quality_score",
        "max_movement_value",
        "avg_movement_value",
        "completed_reps_total",
        "max_right_value",
        "max_left_value",
        "right_reps",
        "left_reps",
        "symmetry_score",
        "valid_angle_ratio",
        "mean_confidence",
        "low_confidence_ratio",
        "angle_range_frames",
        "angle_mean_frames",
        "angle_std_frames",
        "early_angle_mean",
        "late_angle_mean",
        "angle_trend_delta",
        "rep_speed",
        "complete_frame_ratio",
        "target_hit_ratio",
        "rep_completion_ratio",
        "right_left_gap",
    ]

    categorical_features = [
        "patient_name",
        "exercise_code",
        "model_name",
        "movement_name",
    ]

    numeric_features = [c for c in feature_cols if c not in categorical_features]

    X = df[feature_cols]
    y = df["pain_after"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = create_pipeline(categorical_features, numeric_features)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    preds = np.clip(preds, 0.0, 10.0)

    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_cols": feature_cols,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "version": "v2_real_logs",
        },
        MODEL_PATH,
    )

    print(f"[OK] Model kaydedildi: {MODEL_PATH}")
    print(f"[METRIC] MAE : {mae:.3f}")
    print(f"[METRIC] RMSE: {rmse:.3f}")
    print(f"[METRIC] R2  : {r2:.3f}")

    results_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": np.round(preds, 2),
        "abs_error": np.round(np.abs(y_test.values - preds), 2),
    })

    print("\n[TEST ÖRNEKLERİ]")
    print(results_df.head(10).to_string(index=False))

    try:
        rf = pipeline.named_steps["model"]
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = rf.feature_importances_

        pairs = list(zip(feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)

        print("\n[TOP FEATURE IMPORTANCE]")
        for name, score in pairs[:15]:
            print(f"{name}: {score:.4f}")
    except Exception as e:
        print("[WARN] Feature importance alınamadı:", e)


if __name__ == "__main__":
    train()