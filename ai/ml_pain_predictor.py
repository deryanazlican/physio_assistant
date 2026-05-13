import os
import joblib
import pandas as pd


class MLPainPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False

        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                print(f"ML model yüklendi: {self.model_path}")
            except Exception as e:
                print(f"ML model yüklenemedi: {e}")

    def predict_pain_after_exercise(self, current_data, history=None):
        if not self.is_loaded or self.model is None:
            raise RuntimeError("ML model yüklenmemiş.")

        row = {
            "exercise": current_data.get("exercise", ""),
            "angle": current_data.get("angle", 0),
            "quality": current_data.get("quality", 0),
            "reps": current_data.get("reps", 0),
            "duration": current_data.get("duration", 0),
            "current_pain": current_data.get("current_pain", 0),
            "last_exercise_hours_ago": current_data.get("last_exercise_hours_ago", 48.0),
        }

        df = pd.DataFrame([row])
        pred = float(self.model.predict(df)[0])

        pred = max(0.0, min(10.0, pred))

        if pred < 4:
            risk_level = "Düşük"
            risk_color = "🟢"
        elif pred < 7:
            risk_level = "Orta"
            risk_color = "🟡"
        else:
            risk_level = "Yüksek"
            risk_color = "🔴"

        recommendations = []
        warnings = []

        # Confidence (basit ama etkili)
        confidence = 1.0

        if current_data["quality"] < 0.7:
            confidence -= 0.2

        if current_data["last_exercise_hours_ago"] < 6:
            confidence -= 0.2

        if current_data["reps"] < 5:
            confidence -= 0.1

        confidence = max(0.3, min(1.0, confidence))

        if risk_level == "Yüksek":
            warnings.append("Egzersiz sonrası ağrı yüksek görünüyor.")
            recommendations.append("Egzersiz yoğunluğunu azaltın ve dinlenme süresi ekleyin.")
        elif risk_level == "Orta":
            warnings.append("Orta düzey ağrı riski var.")
            recommendations.append("Kontrollü devam edin, formu dikkatle koruyun.")
        else:
            recommendations.append("Ağrı riski düşük. Kontrollü şekilde devam edebilirsiniz.")

        return {
            "predicted_pain": round(pred, 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "warnings": warnings,
            "recommendations": recommendations,
            "confidence": round(confidence, 2)
        }

    def should_continue(self, prediction):
        return prediction.get("predicted_pain", 10) < 7

    def get_recommendation_text(self, prediction):
        pain = prediction.get("predicted_pain", "?")
        risk = prediction.get("risk_level", "?")
        recs = prediction.get("recommendations", [])

        text = f"Tahmini egzersiz sonrası ağrı: {pain}/10. Risk seviyesi: {risk}."
        if recs:
            text += f" Öneri: {recs[0]}"
        return text