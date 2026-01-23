# core/voice_assistant.py
import pyttsx3
import threading
import time
from queue import Queue

class VoiceAssistant:
    """
    Gerçek zamanlı Türkçe sesli geri bildirim sistemi
    """
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.engine = None
        self.speech_queue = Queue()
        self.is_speaking = False
        self.last_message = ""
        self.last_message_time = 0
        self.message_cooldown = 3  # Aynı mesajı 3 saniyede bir söyle
        
        if self.enabled:
            self._initialize_engine()
            self._start_worker()
    
    def _initialize_engine(self):
        """TTS motorunu başlat"""
        try:
            self.engine = pyttsx3.init()
            
            # Türkçe ses ayarları
            voices = self.engine.getProperty('voices')
            # Windows'ta Türkçe ses varsa seç
            for voice in voices:
                if 'turkish' in voice.name.lower() or 'türkçe' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            
            # Hız ve ses seviyesi
            self.engine.setProperty('rate', 160)  # Konuşma hızı
            self.engine.setProperty('volume', 0.9)  # Ses seviyesi
            
            print("✅ Sesli asistan hazır!")
        except Exception as e:
            print(f"⚠️ TTS başlatılamadı: {e}")
            self.enabled = False
    
    def _start_worker(self):
        """Arka plan thread'i başlat"""
        worker = threading.Thread(target=self._speech_worker, daemon=True)
        worker.start()
    
    def _speech_worker(self):
        """Kuyruktan mesaj al ve konuş"""
        while True:
            if not self.speech_queue.empty():
                message = self.speech_queue.get()
                self.is_speaking = True
                
                try:
                    self.engine.say(message)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"Konuşma hatası: {e}")
                
                self.is_speaking = False
            
            time.sleep(0.1)
    
    def speak(self, message, priority=False):
        """
        Mesajı seslendir
        
        Args:
            message: Söylenecek metin
            priority: True ise hemen söyle, False ise cooldown kontrol et
        """
        if not self.enabled or not message:
            return
        
        # Aynı mesajı tekrar etme kontrolü
        current_time = time.time()
        if not priority:
            if message == self.last_message:
                if current_time - self.last_message_time < self.message_cooldown:
                    return  # Çok erken, söyleme
        
        self.last_message = message
        self.last_message_time = current_time
        
        # Kuyruğa ekle
        if priority:
            # Öncelikli mesajlar için kuyruğu temizle
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except:
                    break
        
        self.speech_queue.put(message)
    
    def speak_instruction(self, exercise_name):
        """Egzersiz talimatını söyle"""
        instructions = {
            "ROM_LAT": "Kulaklarınızı omuzlarınıza yaklaştırın",
            "ROM_ROT": "Başınızı sağa ve sola çevirin",
            "ROM_FLEKS": "Çenenizi göğsünüze getirin, sonra tavana bakın",
            "IZO_FLEKS": "Ellerinizi alnınıza koyun ve öne itin",
            "OMUZ_YANA_ACMA": "Sopayı yana kaldırın",
            "DIZ_HAVLU_EZME": "Dizinizin altındaki havluyu ezin",
            # Daha fazla eklenebilir...
        }
        
        if exercise_name in instructions:
            self.speak(instructions[exercise_name], priority=True)
    
    def celebrate(self, achievement):
        """Başarı kutlaması"""
        celebrations = {
            "set_complete": "Harika! Set tamamlandı!",
            "exercise_complete": "Mükemmel! Egzersiz tamamlandı!",
            "perfect_form": "Süper! Formunuz mükemmel!",
            "milestone": "Tebrikler! Yeni bir kilometre taşı!"
        }
        
        if achievement in celebrations:
            self.speak(celebrations[achievement], priority=True)
    
    def warn(self, warning_type):
        """Uyarı mesajları"""
        warnings = {
            "form_error": "Dikkat! Formunuzu düzeltin",
            "too_fast": "Daha yavaş yapın",
            "not_complete": "Hareketi tamamlayın",
            "return_center": "Merkeze dönün"
        }
        
        if warning_type in warnings:
            self.speak(warnings[warning_type])
    
    def count_rep(self, count, total):
        """Tekrar sayısını söyle"""
        self.speak(f"{count}")
    
    def countdown(self, seconds):
        """Geri sayım"""
        if seconds <= 3:
            self.speak(str(seconds), priority=True)
    
    def toggle(self):
        """Sesi aç/kapat"""
        self.enabled = not self.enabled
        status = "açık" if self.enabled else "kapalı"
        print(f"🔊 Ses: {status}")
        return self.enabled


# ==================== KULLANIM ÖRNEĞİ ====================
if __name__ == "__main__":
    # Test
    va = VoiceAssistant(enabled=True)
    
    print("Test başlıyor...")
    time.sleep(1)
    
    va.speak_instruction("ROM_LAT")
    time.sleep(3)
    
    va.count_rep(5, 10)
    time.sleep(2)
    
    va.celebrate("set_complete")
    time.sleep(2)
    
    va.warn("form_error")
    time.sleep(2)
    
    print("Test tamamlandı!")