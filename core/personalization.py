from typing import Dict, List, Any


def get_patient_history(history: List[Dict[str, Any]], patient_name: str) -> List[Dict[str, Any]]:
    result = []
    for item in history:
        if item.get("patient_name") == patient_name:
            result.append(item)
    return result

def generate_adaptive_recommendation(profile: Dict[str, Any], current_prediction: Dict[str, Any]) -> str:
    predicted_pain = float(current_prediction.get("predicted_pain", 0))
    avg_pain_after = float(profile.get("avg_pain_after", 0))
    avg_quality = float(profile.get("avg_quality", 0))
    total_sessions = int(profile.get("total_sessions", 0))

    if total_sessions < 3:
        return "Kişisel profil oluşturuluyor. Birkaç seans daha sonrası için daha doğru öneriler verilecektir."

    if predicted_pain > avg_pain_after + 1:
        return "Bugünkü tahmini ağrı, kişisel ortalamanızın üzerinde. Egzersiz yoğunluğunu azaltmanız önerilir."

    if avg_quality >= 0.85 and predicted_pain <= avg_pain_after:
        return "Formunuz genel olarak iyi. Kontrollü şekilde egzersiz seviyesi artırılabilir."

    if avg_quality < 0.70:
        return "Geçmiş verilere göre form kaliteniz düşük seyrediyor. Önce hareket doğruluğuna odaklanın."

    return "Kişisel geçmişinize göre mevcut egzersiz seviyesi dengeli görünüyor. Kontrollü devam edebilirsiniz."

def build_patient_profile(history: List[Dict[str, Any]], patient_name: str) -> Dict[str, Any]:
    patient_records = get_patient_history(history, patient_name)

    if not patient_records:
        return {
            "patient_name": patient_name,
            "total_sessions": 0,
            "avg_pain_before": 0.0,
            "avg_pain_after": 0.0,
            "avg_quality": 0.0,
            "avg_duration": 0.0,
            "baseline_risk": "unknown"
        }

    pain_before_list = []
    pain_after_list = []
    quality_list = []
    duration_list = []

    for item in patient_records:
        data = item.get("data", {}) or {}

        if data.get("current_pain") is not None:
            pain_before_list.append(float(data.get("current_pain", 0)))

        if data.get("pain_after") is not None:
            pain_after_list.append(float(data.get("pain_after", 0)))

        if data.get("quality") is not None:
            quality_list.append(float(data.get("quality", 0)))

        if data.get("duration") is not None:
            duration_list.append(float(data.get("duration", 0)))

    def avg(values):
        return round(sum(values) / len(values), 2) if values else 0.0

    avg_pain_after = avg(pain_after_list)

    if avg_pain_after < 4:
        baseline_risk = "low"
    elif avg_pain_after < 7:
        baseline_risk = "medium"
    else:
        baseline_risk = "high"

    return {
        "patient_name": patient_name,
        "total_sessions": len(patient_records),
        "avg_pain_before": avg(pain_before_list),
        "avg_pain_after": avg_pain_after,
        "avg_quality": avg(quality_list),
        "avg_duration": avg(duration_list),
        "baseline_risk": baseline_risk
    }