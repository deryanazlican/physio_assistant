from utils.chat_helpers import plan_to_text, session_summary_text
from utils.chat_panel import add_chat_message


class ChatCoordinator:
    def __init__(self, chatbot, planner, session_manager, patient_name: str):
        self.chatbot = chatbot
        self.planner = planner
        self.session_manager = session_manager
        self.patient_name = patient_name

        self.messages = []
        self.current_session = None
        self.current_plan = None

        self.load_existing_data()

    def load_existing_data(self):
        existing_plan = self.planner.load_plan(self.patient_name)
        active_session = self.session_manager.load_active_session(self.patient_name)

        self.current_session = active_session
        self.current_plan = None

        # Sadece condition uyuyorsa eski planı kullan
        if existing_plan and active_session:
            if existing_plan.get("condition") == active_session.get("condition"):
                self.current_plan = existing_plan
        elif existing_plan and not active_session:
            self.current_plan = existing_plan

        if self.current_plan:
            add_chat_message(
                self.messages,
                "assistant",
                "Kayıtlı egzersiz planınız bulundu.\n\n" + plan_to_text(self.current_plan)
            )
        else:
            add_chat_message(
                self.messages,
                "assistant",
                f"Merhaba {self.patient_name}. Şikayetinizi yazın, size uygun egzersiz planı oluşturalım."
            )

        if active_session:
            add_chat_message(
                self.messages,
                "assistant",
                "Devam eden oturumunuz yüklendi. Kaldığınız yerden devam edebilirsiniz.\n\n"
                + session_summary_text(active_session)
            )

    def ensure_session(self, complaint: str, condition: str):
        if self.current_session is None:
            self.current_session = self.session_manager.start_session(
                patient_name=self.patient_name,
                complaint=complaint,
                condition=condition,
                plan={}
            )
        else:
            old_condition = self.current_session.get("condition")
            self.current_session["complaint"] = complaint
            self.current_session["condition"] = condition

            # Condition değiştiyse eski planı iptal et
            if old_condition != condition:
                print(f"[CHAT] condition changed: {old_condition} -> {condition}")
                self.current_plan = None
                self.current_session["plan"] = {}

        self.session_manager.save_active_session(self.current_session)

    def create_plan_for_condition(self, condition: str, complaint: str):
        print(f"[CHAT] create_plan_for_condition condition = {condition}")

        self.ensure_session(complaint, condition)

        # Eğer eldeki plan başka condition'a aitse çöpe at
        if self.current_plan and self.current_plan.get("condition") != condition:
            print(
                f"[CHAT] old plan discarded: "
                f"{self.current_plan.get('condition')} != {condition}"
            )
            self.current_plan = None
            self.current_session["plan"] = {}

        plan = self.planner.create_plan(
            patient_name=self.patient_name,
            condition=condition,
            fitness_level=5,
            weeks=2
        )

        self.current_plan = plan
        self.session_manager.update_plan(self.current_session, plan)
        self.session_manager.save_active_session(self.current_session)

        return plan

    def handle_user_message(self, text: str):
        text = (text or "").strip()
        if not text:
            return

        add_chat_message(self.messages, "user", text)

        analysis = self.chatbot.analyze_message(text)
        condition = analysis["condition"]
        answer = analysis["answer"]

        text_lower = text.lower()

        body_part_keywords = [
            "boyun", "omuz", "diz", "kalça", "kalca", "bel"
        ]

        pain_keywords = [
            "ağrı", "agri", "ağrıyor", "agriyor", "acı", "sızı", "sizi"
        ]

        mentions_body_part = any(k in text_lower for k in body_part_keywords)
        mentions_pain = any(k in text_lower for k in pain_keywords)

        # Sadece plan isteme / onay mesajıysa mevcut condition'ı koru
        if self.current_session is not None:
            old_condition = self.current_session.get("condition")

            is_followup_plan_request = analysis.get("wants_plan") and not mentions_body_part

            if is_followup_plan_request and old_condition:
                print(f"[CHAT] preserving previous condition: {old_condition}")
                condition = old_condition

        print(f"[CHAT] message = {text}")
        print(f"[CHAT] analyzed condition = {condition}")
        print(f"[CHAT] current_session before ensure = {self.current_session.get('condition') if self.current_session else None}")
        print(f"[CHAT] current_plan before ensure = {self.current_plan.get('condition') if self.current_plan else None}")

        self.ensure_session(text, condition)

        print(f"[CHAT] current_session after ensure = {self.current_session.get('condition') if self.current_session else None}")
        print(f"[CHAT] current_plan after ensure = {self.current_plan.get('condition') if self.current_plan else None}")

        if analysis["wants_report"]:
            report_text = session_summary_text(self.current_session)
            add_chat_message(self.messages, "assistant", report_text)
            self.session_manager.save_active_session(self.current_session)
            return

        if analysis["wants_plan"]:
            plan = self.create_plan_for_condition(condition, text)
            combined = answer + "\n\nOluşturulan plan:\n" + plan_to_text(plan)
            add_chat_message(self.messages, "assistant", combined)
            return

        add_chat_message(self.messages, "assistant", answer)
        self.session_manager.save_active_session(self.current_session)

    def show_saved_plan(self):
        if self.current_session is None:
            add_chat_message(self.messages, "assistant", "Önce şikayetinizi yazın, sonra plan oluşturayım.")
            return

        session_condition = self.current_session.get("condition")
        complaint = self.current_session.get("complaint", "")

        print(f"[CHAT] show_saved_plan session_condition = {session_condition}")
        print(f"[CHAT] show_saved_plan current_plan = {self.current_plan.get('condition') if self.current_plan else None}")

        # Plan yoksa veya condition uyuşmuyorsa yeniden üret
        if self.current_plan is None or self.current_plan.get("condition") != session_condition:
            print("[CHAT] regenerating plan from session condition")
            plan = self.planner.create_plan(
                patient_name=self.patient_name,
                condition=session_condition,
                fitness_level=5,
                weeks=2
            )
            self.current_plan = plan
            self.session_manager.update_plan(self.current_session, plan)
            self.session_manager.save_active_session(self.current_session)

        add_chat_message(self.messages, "assistant", plan_to_text(self.current_plan))

    def show_current_report(self):
        if self.current_session:
            add_chat_message(self.messages, "assistant", session_summary_text(self.current_session))
        else:
            add_chat_message(self.messages, "assistant", "Aktif oturum bulunamadı.")

    def add_exercise_result(self, exercise_code: str, target_reps: int, completed_reps: int, duration_sec: int, status="done"):
        if self.current_session is None:
            condition = "BOYUN_AGRISI"
            self.current_session = self.session_manager.start_session(
                patient_name=self.patient_name,
                complaint="Uygulama içinden başlatıldı",
                condition=condition,
                plan=self.current_plan or {}
            )

        self.session_manager.add_exercise_result(
            session=self.current_session,
            exercise_code=exercise_code,
            target_reps=target_reps,
            completed_reps=completed_reps,
            duration_sec=duration_sec,
            status=status,
        )
        self.session_manager.save_active_session(self.current_session)

    def add_note(self, note: str):
        if self.current_session is not None:
            self.session_manager.add_note(self.current_session, note)
            self.session_manager.save_active_session(self.current_session)