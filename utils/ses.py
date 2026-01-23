# utils/ses.py
# v3.0 - HASSASİYETİ DÜZELTİLMİŞ SÜRÜM (SAĞIRLIK GİDERİLDİ)

import pyttsx3
import speech_recognition as sr
import threading
import time

class SesliAsistan:
    def __init__(self):
        # Konuşma Motoru
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 145)
        self.engine.setProperty('volume', 1.0)
        
        # Dinleme Motoru
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        
        # --- AYARLARI GERİ ALDIK (OTOMATİK MOD) ---
        # Burası 3000'di, şimdi 300 yapıp otomatiği açtık.
        self.recognizer.energy_threshold = 300  
        self.recognizer.dynamic_energy_threshold = True # Ortama göre kendini ayarlasın
        self.recognizer.pause_threshold = 0.8 # Cümle bitişini bekleme süresi
        
        self.son_konusulan = ""
        self.son_konusma_zamani = 0
        self.durum = "Mikrofon Başlatılıyor..." 

    def konus(self, metin):
        """Metni sesli okur."""
        if metin == self.son_konusulan and (time.time() - self.son_konusma_zamani) < 4: return
        self.son_konusulan = metin
        self.son_konusma_zamani = time.time()
        
        # Konuşurken durumu güncelleme (Dinleme yazısı bozulmasın diye)
        def _run():
            try:
                self.engine.say(metin)
                self.engine.runAndWait()
            except: pass
        threading.Thread(target=_run, daemon=True).start()

    def komut_dinle_arkaplan(self, callback_fonksiyonu):
        def _listen_loop():
            with self.mic as source:
                self.durum = "🎤 Kalibrasyon..."
                print(self.durum)
                try:
                    # Gürültü ayarı için 1 saniye ortamı dinler
                    self.recognizer.adjust_for_ambient_noise(source, duration=1) 
                except: pass
                
                self.durum = "👂 Seni Dinliyorum..."
                print(f"Hazır! Eşik Değeri: {self.recognizer.energy_threshold}")
                
                while True:
                    try:
                        # 1. DİNLEME
                        audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                        self.durum = "⏳ Algılanıyor..."
                        
                        # 2. ANLAMA
                        try:
                            komut = self.recognizer.recognize_google(audio, language="tr-TR").lower()
                            print(f"🎤 DUYULAN: '{komut}'")
                            
                            # 3. FİLTRELEME (Gereksiz sesleri ele)
                            kelimeler = ["boyun", "omuz", "diz", "kalça", "kalca", "bel", "başla", "geri", "çık", "tamam"]
                            
                            if any(k in komut for k in kelimeler):
                                self.durum = f"✅ Anlaşıldı: {komut}"
                                callback_fonksiyonu(komut)
                                time.sleep(1.5)
                            else:
                                # Alakasız ses duyduysa çaktırma, dinlemeye devam et
                                pass 

                        except sr.UnknownValueError:
                            # Ses anlaşılmadıysa sessizce devam et
                            pass
                        except sr.RequestError:
                            self.durum = "⚠️ İnternet Yok"
                            time.sleep(2)

                    except Exception as e:
                        print(f"Hata: {e}")
                        time.sleep(0.5)
                    
                    # Döngü sonunda mesajı sıfırla
                    if "Anlaşıldı" in self.durum or "Algılanıyor" in self.durum:
                         self.durum = "👂 Seni Dinliyorum..."

        threading.Thread(target=_listen_loop, daemon=True).start()