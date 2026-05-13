def plan_to_text(plan: dict) -> str:
    if not plan:
        return "Kayıtlı plan bulunamadı."

    lines = []
    lines.append(f"Hasta: {plan.get('patient_name', '-')}")
    lines.append(f"Durum: {plan.get('condition', '-')}")
    lines.append(f"Süre: {plan.get('weeks', 0)} hafta")
    lines.append("Plan:")

    schedule = plan.get("schedule", {})
    if not schedule:
        lines.append("Plan takvimi bulunamadı.")
        return "\n".join(lines)

    for week_key, week_data in schedule.items():
        lines.append(f"{week_key}:")
        for day, day_data in week_data.items():
            exercises = ", ".join(day_data.get("exercises", []))
            completed = "✅" if day_data.get("completed") else "⬜"
            lines.append(f"{completed} {day}: {exercises}")

    return "\n".join(lines)


def session_summary_text(session: dict) -> str:
    if not session:
        return "Aktif oturum bulunamadı."

    lines = []
    lines.append(f"Hasta: {session.get('patient_name', '-')}")
    lines.append(f"Şikayet: {session.get('complaint', '-')}")
    lines.append(f"Kategori: {session.get('condition', '-')}")
    lines.append("Yapılanlar:")

    results = session.get("exercise_results", [])
    if not results:
        lines.append("Henüz egzersiz kaydı yok.")
    else:
        for r in results:
            lines.append(
                f"- {r.get('exercise_code', '-')}: "
                f"{r.get('completed_reps', 0)}/{r.get('target_reps', 0)} tekrar, "
                f"{r.get('duration_sec', 0)} sn"
            )

    notes = session.get("notes", [])
    if notes:
        lines.append("Notlar:")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)