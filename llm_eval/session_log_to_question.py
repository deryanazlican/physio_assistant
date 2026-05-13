def session_log_to_question(log_data: dict) -> str:
    summary = log_data.get("summary", {})

    exercise_code = log_data.get("exercise_code", "egzersiz")
    movement_name = summary.get("movement_name", "hareket")

    reps = summary.get("completed_reps", 0)
    target_reps = log_data.get("target_reps", "?")

    avg_angle = summary.get("avg_angle")
    max_angle = summary.get("max_angle")
    target_angle = log_data.get("target_angle")

    quality = summary.get("quality_score")
    symmetry = summary.get("symmetry_score")

    pain = summary.get("pain_before")

    question = f"""
Ben {movement_name} egzersizi yaptım.

Toplam tekrar: {reps}/{target_reps}
Hedef açı: {target_angle}°
Ortalama açı: {avg_angle:.2f}°
Maksimum açı: {max_angle:.2f}°

Kalite skoru: {quality}
Simetri skoru: {symmetry}
Başlangıç ağrı skoru: {pain}

Bu performansa göre bana kısa bir değerlendirme yap ve neyi yanlış yapıyor olabileceğimi söyle.
""".strip()

    return question

def quick_rule_score(answer: str) -> dict:
    score = 0

    if "açı" in answer or "hareket" in answer:
        score += 1
    if "düşük" in answer or "yetersiz" in answer:
        score += 1
    if "öner" in answer or "dikkat" in answer:
        score += 1
    if len(answer) < 300:
        score += 1

    return {
        "score": score,
        "max": 4
    }