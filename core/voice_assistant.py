import pyttsx3
import threading
import time
from queue import Queue, Empty, Full

import pythoncom


class VoiceAssistant:
    """
    Gerçek zamanlı Türkçe sesli geri bildirim sistemi
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.engine = None

        self.speech_queue = Queue(maxsize=20)

        self.is_speaking = False
        self.last_message = ""
        self.last_message_time = 0
        self.message_cooldown = 1.0

        self._worker_started = False
        self._engine_lock = threading.Lock()

        if self.enabled:
            self._start_worker()

    def _initialize_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 160)
            self.engine.setProperty("volume", 1.0)

            voices = self.engine.getProperty("voices")
            print("Toplam voice:", len(voices))
            for v in voices:
                print("VOICE:", getattr(v, "id", ""), getattr(v, "name", ""), getattr(v, "languages", []))

            print("Aktif voice:", self.engine.getProperty("voice"))
            print("✅ Sesli asistan hazır!")

        except Exception as e:
            print(f"⚠️ TTS başlatılamadı: {e}")
            self.engine = None
            self.enabled = False

    def _start_worker(self):
        if self._worker_started:
            return

        worker = threading.Thread(target=self._speech_worker, daemon=True)
        worker.start()
        self._worker_started = True

    def _speech_worker(self):
        """Tüm TTS işlemleri aynı thread'de yürüsün"""
        pythoncom.CoInitialize()

        try:
            self._initialize_engine()

            while True:
                try:
                    message = self.speech_queue.get(timeout=0.2)
                except Empty:
                    continue

                if not self.enabled:
                    continue

                if not message:
                    continue

                self.is_speaking = True
                print(f"🔊 Konuşuyor: {message}")

                try:
                    if self.engine is None:
                        self._initialize_engine()

                    if self.engine is not None:
                        self.engine.stop()
                        self.engine.say(str(message))
                        self.engine.runAndWait()

                except Exception as e:
                    print(f"⚠️ Konuşma hatası: {e}")
                    try:
                        if self.engine is not None:
                            self.engine.stop()
                    except Exception:
                        pass
                    self.engine = None
                    time.sleep(0.2)
                    self._initialize_engine()

                self.is_speaking = False
        finally:
            pythoncom.CoUninitialize()

    def speak(self, message, priority=False):
        if not self.enabled or not message:
            return

        message = str(message).strip()
        if not message:
            return

        now = time.time()

        if (
            not priority
            and message == self.last_message
            and (now - self.last_message_time < self.message_cooldown)
        ):
            return

        self.last_message = message
        self.last_message_time = now

        if priority:
            while True:
                try:
                    self.speech_queue.get_nowait()
                except Exception:
                    break

        try:
            self.speech_queue.put_nowait(message)
        except Full:
            try:
                _ = self.speech_queue.get_nowait()
            except Exception:
                pass
            try:
                self.speech_queue.put_nowait(message)
            except Exception:
                pass

    def speak_instruction(self, exercise_name):
        instructions = {
            "ROM_LAT": "Kulaklarınızı omuzlarınıza yaklaştırın",
            "ROM_ROT": "Başınızı sağa ve sola çevirin",
            "ROM_FLEKS": "Çenenizi göğsünüze getirin, sonra tavana bakın",
            "IZO_FLEKS": "Ellerinizi alnınıza koyun ve öne itin",
            "OMUZ_YANA_ACMA": "Sopayı yana kaldırın",
            "DIZ_HAVLU_EZME": "Dizinizin altındaki havluyu ezin",
        }
        if exercise_name in instructions:
            self.speak(instructions[exercise_name], priority=True)

    def celebrate(self, achievement):
        celebrations = {
            "set_complete": "Harika! Set tamamlandı!",
            "exercise_complete": "Mükemmel! Egzersiz tamamlandı!",
            "perfect_form": "Süper! Formunuz mükemmel!",
            "milestone": "Tebrikler! Yeni bir kilometre taşı!",
        }
        if achievement in celebrations:
            self.speak(celebrations[achievement], priority=True)

    def warn(self, warning_type):
        warnings = {
            "form_error": "Dikkat! Formunuzu düzeltin",
            "too_fast": "Daha yavaş yapın",
            "not_complete": "Hareketi tamamlayın",
            "return_center": "Merkeze dönün",
        }
        if warning_type in warnings:
            self.speak(warnings[warning_type], priority=True)

    def count_rep(self, count, total):
        self.speak(str(count), priority=True)

    def countdown(self, seconds):
        if seconds <= 3:
            self.speak(str(seconds), priority=True)

    def toggle(self):
        self.enabled = not self.enabled
        status = "açık" if self.enabled else "kapalı"
        print(f"🔊 Ses: {status}")

        if self.enabled and not self._worker_started:
            self._start_worker()

        return self.enabled