from .schemas import ExerciseCase


SYSTEM_PROMPT = """
You are a physiotherapy assistant.
Your task is to analyze exercise performance and provide short, clear, safe feedback.
Rules:
1. Be medically cautious.
2. Do not diagnose diseases.
3. Focus on posture, symmetry, control, pain warning, and exercise quality.
4. Keep the answer concise and helpful.
5. Output in Turkish.
""".strip()


def build_user_prompt(case: ExerciseCase) -> str:
    metrics_text = "\n".join([f"- {k}: {v}" for k, v in case.metrics.items()])

    return f"""
Vaka ID: {case.case_id}
Egzersiz: {case.exercise_name}
Hasta notu: {case.patient_note}

Ölçümler:
{metrics_text}

Beklenen odak:
{case.expected_focus}

Lütfen hastaya kısa, anlaşılır ve güvenli Türkçe geri bildirim ver.
Önce kısa değerlendirme yap, sonra 2-3 öneri ver.
""".strip()