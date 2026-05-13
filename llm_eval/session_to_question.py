from __future__ import annotations


def session_log_to_question(log_data: dict) -> str:
    if not isinstance(log_data, dict):
        raise TypeError(f"log_data dict olmalı, gelen tip: {type(log_data).__name__}")

    summary = log_data.get("summary", {})

    if not isinstance(summary, dict):
        raise TypeError(f"summary dict olmalı, gelen tip: {type(summary).__name__}")

    exercise_code = log_data.get("exercise_code", "egzersiz")
    movement_name = summary.get("movement_name", exercise_code)

    reps = summary.get("completed_reps", 0)
    target_reps = log_data.get("target_reps", "?")

    avg_angle = summary.get("avg_angle")
    max_angle = summary.get("max_angle")
    min_angle = summary.get("min_angle")
    target_angle = log_data.get("target_angle")

    quality = summary.get("quality_score")
    symmetry = summary.get("symmetry_score")
    pain_before = summary.get("pain_before")
    pain_after = summary.get("pain_after")

    right_reps = summary.get("right_reps")
    left_reps = summary.get("left_reps")
    max_right = summary.get("max_right_value")
    max_left = summary.get("max_left_value")
    duration = summary.get("duration_sec")
    completion_rate = summary.get("completion_rate")

    details = []

    if reps is not None and target_reps is not None:
        details.append(f"Toplam tekrar: {reps}/{target_reps}")
    if target_angle is not None:
        details.append(f"Hedef açı: {target_angle}°")
    if avg_angle is not None:
        details.append(f"Ortalama açı: {avg_angle:.2f}°")
    if max_angle is not None:
        details.append(f"Maksimum açı: {max_angle:.2f}°")
    if min_angle is not None:
        details.append(f"Minimum açı: {min_angle:.2f}°")
    if quality is not None:
        details.append(f"Kalite skoru: {quality}")
    if symmetry is not None:
        details.append(f"Simetri skoru: {symmetry}")
    if pain_before is not None:
        details.append(f"Başlangıç ağrı skoru: {pain_before}")
    if pain_after is not None:
        details.append(f"Bitiş ağrı skoru: {pain_after}")
    if right_reps is not None and left_reps is not None:
        details.append(f"Sağ tekrar: {right_reps}, Sol tekrar: {left_reps}")
    if max_right is not None and max_left is not None:
        details.append(f"Sağ maksimum açı: {max_right:.2f}°, Sol maksimum açı: {max_left:.2f}°")
    if duration is not None:
        details.append(f"Süre: {duration:.2f} saniye")
    if completion_rate is not None:
        details.append(f"Tamamlama oranı: {completion_rate}")

    details_text = "\n".join(details)

    question = f"""
Ben {movement_name} egzersizi yaptım.

{details_text}

Bu performansa göre bana kısa bir değerlendirme yap.
Lütfen:
1. Genel performansı kısaca değerlendir
2. Eksik veya zayıf noktayı söyle
3. 2-3 kısa öneri ver
4. Türkçe ve anlaşılır yaz
""".strip()

    return question