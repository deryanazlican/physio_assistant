def extract_case_from_log(log_item):
    data = log_item.get("data", {})

    case = {
        "source_file": log_item.get("file_name"),
        "exercise_name": data.get("exercise_name") or data.get("exercise") or "Bilinmeyen Egzersiz",
        "rep_count": data.get("rep_count") or data.get("reps") or 0,
        "quality_score": data.get("quality_score") or data.get("quality") or None,
        "symmetry_score": data.get("symmetry_score") or data.get("symmetry") or None,
        "risk_level": data.get("risk_level") or data.get("pain_risk") or "Bilinmiyor",
        "patient_name": data.get("patient_name") or data.get("hasta_adi") or "Anonim",
        "metrics": data
    }

    return case

def build_chatbot_style_prompt(case):
    metrics = case.get("metrics", {})

    important_lines = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            important_lines.append(f"- {key}: {value}")

    metrics_text = "\n".join(important_lines[:30])

    prompt = f"""
Sen bir fizyoterapi destek asistanısın.

Aşağıda bir egzersiz seansının gerçek analiz verileri bulunmaktadır.
Lütfen hastaya Türkçe, kısa, anlaşılır ve güvenli bir geri bildirim ver.

Hasta: {case.get("patient_name")}
Egzersiz: {case.get("exercise_name")}
Tekrar sayısı: {case.get("rep_count")}
Kalite skoru: {case.get("quality_score")}
Simetri skoru: {case.get("symmetry_score")}
Risk seviyesi: {case.get("risk_level")}

Seans verileri:
{metrics_text}

Kurallar:
- Teşhis koyma.
- Kısa değerlendirme yap.
- 2 veya 3 somut öneri ver.
- Gerekirse ağrı artarsa uzman görüşü alınmasını belirt.
- Çıktı Türkçe olsun.
""".strip()

    return prompt