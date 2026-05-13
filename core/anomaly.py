def detect_anomaly(profile, current_prediction):
    avg = profile.get("avg_pain_after", 0)
    pred = current_prediction.get("predicted_pain", 0)

    if avg == 0:
        return None

    if pred > avg + 2:
        return "⚠️ Olağandışı yüksek ağrı tespit edildi."

    if pred < avg - 2:
        return "📉 Beklenenden düşük ağrı (iyi gelişme)."

    return None