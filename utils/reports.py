# utils/reports.py

import os
from datetime import datetime


def kaydet(hasta_ismi, veriler):
    if not veriler:
        return None

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "raporlar")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    zaman_damgasi = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dosya_adi = f"Rapor_{hasta_ismi}_{zaman_damgasi}.txt"
    tam_yol = os.path.join(save_dir, dosya_adi)

    icerik = []
    icerik.append(f"HASTA: {hasta_ismi}")
    icerik.append(f"TARIH: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    icerik.append("-" * 30)

    for veri in veriler:
        icerik.append(veri)

    icerik.append("-" * 30)
    icerik.append("FIZYO ASISTAN AI RAPORU SONU")

    try:
        with open(tam_yol, "w", encoding="utf-8") as f:
            f.write("\n".join(icerik))
        print(f"✅ Rapor şuraya kaydedildi: {tam_yol}")
        return tam_yol
    except Exception as e:
        print(f"❌ Rapor kaydetme hatası: {e}")
        return None


def estimate_adherence(session: dict) -> float:
    results = session.get("exercise_results", [])
    if not results:
        return 0.0

    total_target = 0
    total_done = 0

    for r in results:
        total_target += r.get("target_reps", 0)
        total_done += r.get("completed_reps", 0)

    if total_target <= 0:
        return 0.0

    return (total_done / total_target) * 100.0


def generate_progress_comment(adherence: float) -> str:
    if adherence >= 90:
        return "Kullanıcı planın büyük kısmını başarıyla tamamladı. Egzersiz uyumu çok iyi."
    elif adherence >= 70:
        return "Kullanıcı planın önemli bir kısmını tamamladı. Düzenli devam edilirse ilerleme desteklenebilir."
    elif adherence >= 50:
        return "Kullanıcı egzersizlerin bir kısmını tamamladı. Daha düzenli uygulama faydalı olabilir."
    else:
        return "Egzersiz tamamlama düzeyi düşük. Daha kısa veya daha kolay bir plan daha uygun olabilir."


def session_to_lines(session: dict) -> list[str]:
    lines = []

    patient_name = session.get("patient_name", "Bilinmiyor")
    complaint = session.get("complaint", "Belirtilmedi")
    condition = session.get("condition", "Belirtilmedi")
    created_at = session.get("created_at", "")
    ended_at = session.get("ended_at", "")

    lines.append(f"HASTA: {patient_name}")
    lines.append(f"ŞİKAYET: {complaint}")
    lines.append(f"KATEGORİ: {condition}")
    lines.append(f"BAŞLANGIÇ: {created_at}")
    lines.append(f"BİTİŞ: {ended_at}")
    lines.append("-" * 40)

    lines.append("OLUŞTURULAN PLAN:")
    plan = session.get("plan", {})
    schedule = plan.get("schedule", {})

    for week_key, week_data in schedule.items():
        lines.append(f"{week_key}:")
        for day, day_data in week_data.items():
            exs = ", ".join(day_data.get("exercises", []))
            lines.append(f"  {day}: {exs}")

    lines.append("-" * 40)
    lines.append("OTURUMDA YAPILANLAR:")

    results = session.get("exercise_results", [])
    if not results:
        lines.append("Herhangi bir egzersiz kaydı bulunamadı.")
    else:
        for r in results:
            lines.append(
                f"{r.get('exercise_code', '-')}"
                f" | hedef: {r.get('target_reps', 0)}"
                f" | yapilan: {r.get('completed_reps', 0)}"
                f" | sure(sn): {r.get('duration_sec', 0)}"
                f" | durum: {r.get('status', '-')}"
            )

    lines.append("-" * 40)
    adherence = estimate_adherence(session)
    lines.append(f"TAHMİNİ PLAN UYUMU: %{adherence:.1f}")
    lines.append(generate_progress_comment(adherence))

    notes = session.get("notes", [])
    if notes:
        lines.append("-" * 40)
        lines.append("NOTLAR:")
        for note in notes:
            lines.append(f"- {note}")

    return lines


def kaydet_session_raporu(session: dict):
    hasta_ismi = session.get("patient_name", "Bilinmiyor")
    veriler = session_to_lines(session)
    return kaydet(hasta_ismi, veriler)