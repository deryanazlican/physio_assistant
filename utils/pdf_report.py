from fpdf import FPDF
from datetime import datetime
import os


class SessionPDF(FPDF):
    def header(self):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 10, "Physio Assistant Session Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", size=9)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _safe_text(text):
    if text is None:
        return ""
    return str(text)


def _register_unicode_fonts(pdf: FPDF):
    windows_font_dir = r"C:\Windows\Fonts"

    regular_candidates = [
        os.path.join(windows_font_dir, "DejaVuSans.ttf"),
        os.path.join(windows_font_dir, "arial.ttf"),
        os.path.join(windows_font_dir, "segoeui.ttf"),
    ]
    bold_candidates = [
        os.path.join(windows_font_dir, "DejaVuSans-Bold.ttf"),
        os.path.join(windows_font_dir, "arialbd.ttf"),
        os.path.join(windows_font_dir, "segoeuib.ttf"),
    ]

    regular_font = next((p for p in regular_candidates if os.path.exists(p)), None)
    bold_font = next((p for p in bold_candidates if os.path.exists(p)), None)

    if not regular_font:
        raise FileNotFoundError("Unicode font bulunamadı.")

    pdf.add_font("DejaVu", "", regular_font)
    if bold_font:
        pdf.add_font("DejaVu", "B", bold_font)
    else:
        pdf.add_font("DejaVu", "B", regular_font)


def write_line(pdf: FPDF, text, h=7):
    pdf.set_x(pdf.l_margin)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(usable_width, h, str(text))


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


def export_session_pdf(session: dict, output_path: str):
    pdf = SessionPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    _register_unicode_fonts(pdf)
    pdf.add_page()
    pdf.set_font("DejaVu", size=11)

    patient_name = _safe_text(session.get("patient_name", "Bilinmiyor"))
    complaint = _safe_text(session.get("complaint", "Belirtilmedi"))
    condition = _safe_text(session.get("condition", "Belirtilmedi"))
    created_at = _safe_text(session.get("created_at", ""))
    ended_at = _safe_text(session.get("ended_at", ""))

    pdf.set_font("DejaVu", "B", 12)
    write_line(pdf, "Oturum Bilgileri", 8)
    pdf.set_font("DejaVu", size=11)

    write_line(pdf, f"Hasta Adı: {patient_name}")
    write_line(pdf, f"Şikayet: {complaint}")
    write_line(pdf, f"Kategori: {condition}")
    write_line(pdf, f"Başlangıç: {created_at}")
    write_line(pdf, f"Bitiş: {ended_at}")
    pdf.ln(3)

    pdf.set_font("DejaVu", "B", 12)
    write_line(pdf, "Oluşturulan Plan", 8)
    pdf.set_font("DejaVu", size=11)

    plan = session.get("plan", {})
    schedule = plan.get("schedule", {})

    if not schedule:
        write_line(pdf, "Plan verisi bulunamadı.")
    else:
        for week_key, week_data in schedule.items():
            pdf.set_font("DejaVu", "B", 11)
            write_line(pdf, week_key)
            pdf.set_font("DejaVu", size=11)

            for day, day_data in week_data.items():
                exercises = ", ".join(day_data.get("exercises", []))
                write_line(pdf, f"{day}: {exercises}")
            pdf.ln(1)

    pdf.ln(2)
    pdf.set_font("DejaVu", "B", 12)
    write_line(pdf, "Yapılan Egzersizler", 8)
    pdf.set_font("DejaVu", size=11)

    results = session.get("exercise_results", [])
    if not results:
        write_line(pdf, "Herhangi bir egzersiz kaydı bulunamadı.")
    else:
        for r in results:
            write_line(pdf, f"Egzersiz: {r.get('exercise_code', '-')}")
            write_line(
                pdf,
                f"Hedef Tekrar: {r.get('target_reps', 0)} | "
                f"Yapılan Tekrar: {r.get('completed_reps', 0)} | "
                f"Süre(sn): {r.get('duration_sec', 0)} | "
                f"Durum: {r.get('status', '-')}"
            )
            pdf.ln(1)

    adherence = estimate_adherence(session)
    comment = generate_progress_comment(adherence)

    pdf.ln(2)
    pdf.set_font("DejaVu", "B", 12)
    write_line(pdf, "İlerleme Özeti", 8)
    pdf.set_font("DejaVu", size=11)
    write_line(pdf, f"Tahmini Plan Uyumu: %{adherence:.1f}")
    write_line(pdf, f"Yorum: {comment}")

    notes = session.get("notes", [])
    if notes:
        pdf.ln(2)
        pdf.set_font("DejaVu", "B", 12)
        write_line(pdf, "Notlar", 8)
        pdf.set_font("DejaVu", size=11)
        for note in notes:
            write_line(pdf, f"- {note}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)


def export_session_pdf_auto(session: dict, folder="raporlar_pdf") -> str:
    patient_name = str(session.get("patient_name", "Bilinmiyor")).replace(" ", "_")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = os.path.join(folder, f"Rapor_{patient_name}_{timestamp}.pdf")
    export_session_pdf(session, output_path)
    return output_path