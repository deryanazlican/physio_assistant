# ai/chatbot.py
from __future__ import annotations

from ai.llm_backends import GeminiBackend, TogetherBackend, OllamaBackend


class PhysioChatbot:
    def __init__(
        self,
        backend_type: str = "gemini",
        api_key: str | None = None,
        model_name: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.enabled = False
        self.backend = None
        self.backend_type = backend_type
        self.history: list[str] = []

        default_models = {
            "gemini": "gemini-2.5-flash",
            "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "ollama": "llama3.1",
        }

        self.model_name = model_name or default_models.get(backend_type, "gemini-2.5-flash")

        try:
            if backend_type == "gemini":
                if not api_key:
                    print("⚠️ Gemini API key yok. Chatbot devre dışı.")
                    return
                self.backend = GeminiBackend(api_key=api_key, model_name=self.model_name)

            elif backend_type == "together":
                if not api_key:
                    print("⚠️ Together API key yok. Chatbot devre dışı.")
                    return
                self.backend = TogetherBackend(api_key=api_key, model_name=self.model_name)

            elif backend_type == "ollama":
                self.backend = OllamaBackend(model_name=self.model_name, base_url=ollama_base_url)

            else:
                print(f"⚠️ Desteklenmeyen backend: {backend_type}")
                return

            self.enabled = True
            if False:
                print(f"✅ Chatbot hazır! (backend={self.backend_type}, model={self.model_name})")

        except Exception as e:
            print(f"⚠️ Chatbot başlatma hatası: {e}")
            self.enabled = False

        self.system_prompt = (
            "Sen Türkçe konuşan, destekleyici bir fizyoterapi asistanısın.\n"
            "Kullanıcının yazdığı şikayete göre uygulamadaki egzersizlere yönlendirme yaparsın.\n"
            "Doktor gibi kesin tanı koymazsın.\n"
            "Acil durumlarda doktora yönlendirirsin.\n"
            "Yanıtların kısa, net ve anlaşılır olur.\n"
            "Mümkün olduğunda kullanıcıya egzersiz planı oluşturulabileceğini belirtirsin.\n"
            "100 kelimeyi geçme.\n"
        )

    def detect_condition(self, text: str) -> str:
        t = (text or "").lower()

        if any(k in t for k in ["boyun", "ense", "cervical"]):
            return "BOYUN_AGRISI"
        if any(k in t for k in ["omuz", "shoulder"]):
            return "OMUZ_AGRISI"
        if any(k in t for k in ["diz", "knee"]):
            return "DIZ_AGRISI"
        if any(k in t for k in ["kalça", "kalca", "hip"]):
            return "KALCA_AGRISI"
        if any(k in t for k in ["bel", "lumbar", "back"]):
            return "BEL_AGRISI"

        return "BOYUN_AGRISI"

    def _build_prompt(self, question: str) -> str:
        history_text = ""
        if self.history:
            history_text = "\n".join(self.history[-8:]) + "\n"

        return (
            f"{self.system_prompt}\n"
            f"{history_text}"
            f"Kullanıcı: {question}\n"
            f"Asistan:"
        )

    def ask(self, question: str) -> tuple[str, float]:
        import time

        if not self.enabled or not self.backend:
            return "❌ Chatbot şu an kullanılamıyor.", 0.0

        q = (question or "").strip()
        if len(q) < 3:
            return "Biraz daha detay yazar mısın?", 0.0

        lower = q.lower()
        if any(w in lower for w in ["acil", "şiddetli ağrı", "kan", "kırık", "nefes darlığı", "göğüs ağrısı"]):
            return "⚠️ Bu ciddi olabilir. Lütfen doktora başvur.", 0.0

        prompt = self._build_prompt(q)

        delay = 1.5
        last_error = None

        for _ in range(3):  # 3 deneme
            try:
                if self.backend_type == "gemini":
                    time.sleep(1.0)

                answer, latency = self.backend.generate(prompt)

                if not answer:
                    answer = "❌ Yanıt alınamadı."

                if len(answer) > 700:
                    answer = answer[:700] + "..."

                self.history.append(f"Kullanıcı: {q}")
                self.history.append(f"Asistan: {answer}")

                return answer, latency

            except Exception as e:
                last_error = e
                err = str(e)

                if "503" in err or "UNAVAILABLE" in err or "high demand" in err:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    break

        # retry bitti
        err = str(last_error)

        if "503" in err or "UNAVAILABLE" in err or "high demand" in err:
            return "AI şu anda yoğun. Lütfen birkaç saniye sonra tekrar deneyin.", 0.0

        return "Yanıt alınamadı.", 0.0

    def reset(self) -> str:
        self.history = []
        return "✅ Sohbet geçmişi temizlendi"

    def wants_plan(self, text: str) -> bool:
        t = (text or "").lower()
        keywords = [
            "plan",
            "egzersiz planı",
            "plan istiyorum",
            "ne yapmalıyım",
            "hangi hareketleri yapayım",
            "hareket öner",
            "egzersiz öner",
            "ne yapayım",
            "program oluştur",
            "başlayalım",
        ]
        return any(k in t for k in keywords)

    def is_report_request(self, text: str) -> bool:
        t = (text or "").lower()
        keywords = [
            "raporumu göster",
            "rapor göster",
            "oturumumu göster",
            "neler yaptım",
            "planımı göster",
            "kayıtları göster",
            "bugün ne yapacaktım",
        ]
        return any(k in t for k in keywords)

    def analyze_message(self, question: str) -> dict:
        condition = self.detect_condition(question)
        answer, latency = self.ask(question)

        return {
            "condition": condition,
            "answer": answer,
            "wants_plan": self.wants_plan(question),
            "wants_report": self.is_report_request(question),
            "latency": latency,
            "model_name": self.model_name,
            "backend_type": self.backend_type,
        }