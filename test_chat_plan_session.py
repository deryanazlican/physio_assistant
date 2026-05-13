from ai.chatbot import PhysioChatbot
from core.plan_generator import PersonalizedPlanGenerator
from core.session_manager import SessionManager
from utils.reports import kaydet_session_raporu
from utils.pdf_report import export_session_pdf_auto

API_KEY = "AIzaSyD-hlWo6gMsrzNm8_Z-yopqalazr3BKWYA"

chatbot = PhysioChatbot(api_key=API_KEY)
planner = PersonalizedPlanGenerator()
session_manager = SessionManager()

patient_name = "Birgul"
complaint = "Boynum ağrıyor ve sağa sola çevirince zorlanıyorum."

analysis = chatbot.analyze_and_reply(complaint)
condition = analysis["condition"]

print("Chatbot cevabi:")
print(analysis["answer"])
print("Condition:", condition)

plan = planner.create_plan(
    patient_name=patient_name,
    condition=condition,
    fitness_level=5,
    weeks=2
)

session = session_manager.start_session(
    patient_name=patient_name,
    complaint=complaint,
    condition=condition,
    plan=plan
)

session_manager.add_exercise_result(session, "ROM_LAT", target_reps=10, completed_reps=8, duration_sec=35)
session_manager.add_exercise_result(session, "ROM_ROT", target_reps=10, completed_reps=10, duration_sec=40)
session_manager.add_note(session, "Kullanici rotasyon hareketini daha rahat yapti.")

session_manager.end_session(session)
json_path = session_manager.save_session(session)
txt_path = kaydet_session_raporu(session)
pdf_path = export_session_pdf_auto(session)

print("Session JSON:", json_path)
print("TXT rapor:", txt_path)
print("PDF rapor:", pdf_path)