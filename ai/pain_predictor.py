from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


class SimplePainPredictor:
    def predict_pain_after_exercise(self, current_data: Dict[str, Any], exercise_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        pain_before = _to_float(current_data.get("pain_before", current_data.get("current_pain", 3)), 3.0)
        quality = _to_float(current_data.get("quality_score", current_data.get("quality", 0.7)), 0.7)
        symmetry = _to_float(current_data.get("symmetry_score", 0.7), 0.7)
        completed_reps = _to_int(current_data.get("completed_reps", current_data.get("reps", 0)), 0)
        target_reps = _to_int(current_data.get("target_reps", 10), 10)
        max_angle = _to_float(current_data.get("max_angle", current_data.get("angle", 0)), 0.0)
        target_angle = _to_float(current_data.get("target_angle", 1.0), 1.0)
        duration_sec = _to_float(current_data.get("duration_sec", current_data.get("duration", 0.0)), 0.0)

        predicted_pain = pain_before
        warnings: List[str] = []
        recommendations: List[str] = []

        rep_ratio = (completed_reps / target_reps) if target_reps > 0 else 0.0
        angle_ratio = (max_angle / target_angle) if target_angle > 0 else 0.0

        if quality < 0.60:
            predicted_pain += 1.2
            warnings.append("⚠️ Form kalitesi düşük.")
            recommendations.append("Hareketi daha kontrollü yapın.")

        if symmetry < 0.45:
            predicted_pain += 0.8
            warnings.append("⚠️ Simetri düşük.")
            recommendations.append("Sağ-sol dengesini korumaya odaklanın.")

        if rep_ratio < 0.70:
            predicted_pain += 0.4
            recommendations.append("Hedef tekrar sayısına kontrollü şekilde yaklaşın.")

        if angle_ratio > 1.15:
            predicted_pain += 0.8
            warnings.append("⚠️ Hedef açının üzerine çıkılmış olabilir.")
            recommendations.append("Ağrısız hareket aralığında kalın.")

        if duration_sec > 90:
            predicted_pain += 0.4
            recommendations.append("Egzersizi biraz daha kısa setlere bölebilirsiniz.")

        if quality > 0.85:
            predicted_pain -= 0.3

        if symmetry > 0.80:
            predicted_pain -= 0.2

        predicted_pain = float(np.clip(predicted_pain, 0.0, 10.0))
        return self._build_response(predicted_pain, pain_before, warnings, recommendations, confidence=0.45)

    def _build_response(
        self,
        predicted_pain: float,
        pain_before: float,
        warnings: List[str],
        recommendations: List[str],
        confidence: float,
        top_factors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        pain_delta = predicted_pain - pain_before

        if predicted_pain < 4:
            risk_level = "Düşük"
            risk_color = "🟢"
        elif predicted_pain < 7:
            risk_level = "Orta"
            risk_color = "🟡"
        else:
            risk_level = "Yüksek"
            risk_color = "🔴"

        if not recommendations:
            recommendations = ["Kontrollü şekilde devam edin."]

        return {
            "predicted_pain": round(predicted_pain, 2),
            "predicted_delta": round(pain_delta, 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "warnings": warnings,
            "recommendations": recommendations,
            "confidence": round(float(np.clip(confidence, 0.30, 0.99)), 2),
            "top_factors": top_factors or [],
        }

    def should_continue(self, prediction: Dict[str, Any]) -> bool:
        return _to_float(prediction.get("predicted_pain", 10.0), 10.0) < 7.0

    def get_recommendation_text(self, prediction: Dict[str, Any]) -> str:
        pain = prediction.get("predicted_pain", "?")
        delta = prediction.get("predicted_delta", "?")
        risk = prediction.get("risk_level", "?")
        recs = prediction.get("recommendations", [])
        return f"Tahmini egzersiz sonrası ağrı: {pain}/10 (Δ {delta}). Risk seviyesi: {risk}. " + (f"Öneri: {recs[0]}" if recs else "")


class MLPainPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.bundle = None
        self.model = None
        self.feature_cols: List[str] = []
        self.is_loaded = False
        self.fallback = SimplePainPredictor()

        if os.path.exists(self.model_path):
            try:
                self.bundle = joblib.load(self.model_path)
                self.model = self.bundle["model"]
                self.feature_cols = self.bundle["feature_cols"]
                self.is_loaded = True
                print(f"ML model yüklendi: {self.model_path}")
            except Exception as e:
                print(f"ML model yüklenemedi: {e}")

    def _extract_features(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        features = {
            "patient_name": str(current_data.get("patient_name", "UNKNOWN")),
            "exercise_code": str(current_data.get("exercise_code", current_data.get("exercise", "UNKNOWN"))),
            "model_name": str(current_data.get("model_name", "unknown")),
            "target_angle": _to_float(current_data.get("target_angle", 0.0)),
            "target_reps": _to_int(current_data.get("target_reps", 0)),

            "frame_count": _to_int(current_data.get("frame_count", 0)),
            "completed_reps": _to_int(current_data.get("completed_reps", current_data.get("reps", 0))),
            "duration_sec": _to_float(current_data.get("duration_sec", current_data.get("duration", 0.0))),
            "avg_angle": _to_float(current_data.get("avg_angle", current_data.get("angle", 0.0))),
            "max_angle": _to_float(current_data.get("max_angle", current_data.get("angle", 0.0))),
            "min_angle": _to_float(current_data.get("min_angle", 0.0)),
            "std_angle": _to_float(current_data.get("std_angle", 0.0)),
            "avg_fps": _to_float(current_data.get("avg_fps", 0.0)),
            "min_fps": _to_float(current_data.get("min_fps", 0.0)),
            "max_fps": _to_float(current_data.get("max_fps", 0.0)),
            "completion_frame_count": _to_int(current_data.get("completion_frame_count", 0)),
            "completion_rate": _to_float(current_data.get("completion_rate", 0.0)),
            "pain_before": _to_float(current_data.get("pain_before", current_data.get("current_pain", 0.0))),
            "missing_angle_to_target": _to_float(current_data.get("missing_angle_to_target", 0.0)),
            "movement_name": str(current_data.get("movement_name", "unknown")),
            "movement_value": _to_float(current_data.get("movement_value", 0.0)),
            "movement_target": _to_float(current_data.get("movement_target", 0.0)),
            "quality_score": _to_float(current_data.get("quality_score", current_data.get("quality", 0.0))),
            "max_movement_value": _to_float(current_data.get("max_movement_value", 0.0)),
            "avg_movement_value": _to_float(current_data.get("avg_movement_value", 0.0)),
            "completed_reps_total": _to_int(current_data.get("completed_reps_total", current_data.get("completed_reps", 0))),
            "max_right_value": _to_float(current_data.get("max_right_value", 0.0)),
            "max_left_value": _to_float(current_data.get("max_left_value", 0.0)),
            "right_reps": _to_int(current_data.get("right_reps", 0)),
            "left_reps": _to_int(current_data.get("left_reps", 0)),
            "symmetry_score": _to_float(current_data.get("symmetry_score", 0.0)),

            "valid_angle_ratio": _to_float(current_data.get("valid_angle_ratio", 0.0)),
            "mean_confidence": _to_float(current_data.get("mean_confidence", 0.0)),
            "low_confidence_ratio": _to_float(current_data.get("low_confidence_ratio", 0.0)),
            "angle_range_frames": _to_float(current_data.get("angle_range_frames", 0.0)),
            "angle_mean_frames": _to_float(current_data.get("angle_mean_frames", 0.0)),
            "angle_std_frames": _to_float(current_data.get("angle_std_frames", 0.0)),
            "early_angle_mean": _to_float(current_data.get("early_angle_mean", 0.0)),
            "late_angle_mean": _to_float(current_data.get("late_angle_mean", 0.0)),
            "angle_trend_delta": _to_float(current_data.get("angle_trend_delta", 0.0)),
            "rep_speed": _to_float(current_data.get("rep_speed", 0.0)),
            "complete_frame_ratio": _to_float(current_data.get("complete_frame_ratio", 0.0)),
        }

        target_angle = features["target_angle"]
        target_reps = features["target_reps"]
        features["target_hit_ratio"] = (features["max_angle"] / target_angle) if target_angle > 0 else 0.0
        features["rep_completion_ratio"] = (features["completed_reps"] / target_reps) if target_reps > 0 else 0.0
        features["right_left_gap"] = abs(features["max_right_value"] - features["max_left_value"])

        return features

    def _build_explanations(self, feats: Dict[str, Any]) -> List[str]:
        factors = []

        if _to_float(feats.get("quality_score", 0.0)) < 0.60:
            factors.append("low_quality")
        if _to_float(feats.get("symmetry_score", 1.0)) < 0.45:
            factors.append("low_symmetry")
        if _to_float(feats.get("target_hit_ratio", 0.0)) > 1.15:
            factors.append("over_target_angle")
        if _to_float(feats.get("rep_completion_ratio", 1.0)) < 0.70:
            factors.append("low_rep_completion")
        if _to_float(feats.get("low_confidence_ratio", 0.0)) > 0.40:
            factors.append("unstable_tracking")
        if _to_float(feats.get("angle_trend_delta", 0.0)) < -3.0:
            factors.append("late_session_drop")

        return factors

    def _confidence_score(self, feats: Dict[str, Any]) -> float:
        confidence = 0.88

        if _to_float(feats.get("low_confidence_ratio", 0.0)) > 0.40:
            confidence -= 0.20
        if _to_float(feats.get("valid_angle_ratio", 0.0)) < 0.50:
            confidence -= 0.20
        if _to_int(feats.get("frame_count", 0)) < 60:
            confidence -= 0.10
        if _to_float(feats.get("mean_confidence", 0.0)) < 0.50:
            confidence -= 0.10

        return float(np.clip(confidence, 0.35, 0.98))

    def predict_pain_after_exercise(self, current_data: Dict[str, Any], history=None) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            return self.fallback.predict_pain_after_exercise(current_data, history)

        feats = self._extract_features(current_data)
        df = pd.DataFrame([{col: feats.get(col, None) for col in self.feature_cols}])

        pred = float(self.model.predict(df)[0])
        pred = float(np.clip(pred, 0.0, 10.0))

        pain_before = _to_float(feats.get("pain_before", 0.0), 0.0)
        delta = pred - pain_before

        if pred < 4:
            risk_level = "Düşük"
            risk_color = "🟢"
        elif pred < 7:
            risk_level = "Orta"
            risk_color = "🟡"
        else:
            risk_level = "Yüksek"
            risk_color = "🔴"

        top_factors = self._build_explanations(feats)
        confidence = self._confidence_score(feats)

        warnings: List[str] = []
        recommendations: List[str] = []

        if risk_level == "Yüksek":
            warnings.append("Egzersiz sonrası ağrı yüksek görünüyor.")
            recommendations.append("Egzersiz yoğunluğunu azaltın ve gerekirse ara verin.")
        elif risk_level == "Orta":
            warnings.append("Orta düzey ağrı artışı riski var.")
            recommendations.append("Kontrollü devam edin, formu koruyun.")
        else:
            recommendations.append("Ağrı riski düşük görünüyor, kontrollü devam edebilirsiniz.")

        if "low_quality" in top_factors:
            recommendations.append("Form kalitesini artırmak için hareketi yavaşlatın.")
        if "low_symmetry" in top_factors:
            recommendations.append("Sağ-sol dengeyi iyileştirmeye odaklanın.")
        if "over_target_angle" in top_factors:
            recommendations.append("Hedef açının üzerine çıkmayın.")
        if "late_session_drop" in top_factors:
            recommendations.append("Yorgunluk artmış olabilir, setleri kısaltmayı düşünün.")

        return {
            "predicted_pain": round(pred, 2),
            "predicted_delta": round(delta, 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "warnings": warnings,
            "recommendations": recommendations,
            "confidence": round(confidence, 2),
            "top_factors": top_factors,
        }

    def should_continue(self, prediction: Dict[str, Any]) -> bool:
        return prediction.get("predicted_pain", 10) < 7

    def get_recommendation_text(self, prediction: Dict[str, Any]) -> str:
        pain = prediction.get("predicted_pain", "?")
        delta = prediction.get("predicted_delta", "?")
        risk = prediction.get("risk_level", "?")
        recs = prediction.get("recommendations", [])
        text = f"Tahmini egzersiz sonrası ağrı: {pain}/10. Değişim: {delta}. Risk seviyesi: {risk}."
        if recs:
            text += f" Öneri: {recs[0]}"
        return text