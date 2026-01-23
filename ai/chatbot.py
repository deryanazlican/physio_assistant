"""
Gelişmiş Fizyoterapi Chatbot
Gemini API ile hasta sorularını yanıtlar
"""

import google.generativeai as genai
from config import Config

class PhysioChatbot:
    def __init__(self, api_key=None):
        self.enabled = False
        self.model = None
        self.chat_session = None
        
        if api_key and api_key != "AIzaSyBQUzyGcWm9voPr9vvStpoiW37xBykZka0":
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                
                # İlk sistem talimatı
                system_prompt = """
Sen profesyonel bir fizyoterapistsin. Türkçe konuşuyorsun.
Hastalarının egzersiz, ağrı yönetimi ve rehabilitasyon sorularını yanıtlıyorsun.

ÖNEMLİ KURALLAR:
- Yanıtları 100 kelime ile sınırla
- Tıbbi acil durumlarda doktora yönlendir
- Net, anlaşılır ve destekleyici ol
- Egzersiz önerileri verirken güvenliği ön planda tut
"""
                # Chat başlat
                self.chat_session = self.model.start_chat(history=[])
                response = self.chat_session.send_message(system_prompt)
                
                self.enabled = True
                print("✅ Chatbot hazır!")
                
            except Exception as e:
                print(f"⚠️ Chatbot başlatma hatası: {e}")
                self.enabled = False
        else:
            print("⚠️ Gemini API key yok. Chatbot devre dışı.")
    
    def ask(self, question):
        """Kullanıcı sorusunu yanıtla"""
        if not self.enabled:
            return "❌ Chatbot şu an kullanılamıyor. Lütfen config.py'de GEMINI_API_KEY ayarlayın."
        
        try:
            # Özel durum kontrolü
            if any(word in question.lower() for word in ['acil', 'şiddetli ağrı', 'kan', 'kırık']):
                return "⚠️ Bu ciddi bir durum! Lütfen EN KISA SÜREDE bir sağlık kurumuna başvurun."
            
            # Gemini'ye sor
            response = self.chat_session.send_message(question)
            answer = response.text
            
            # Uzun cevapları kısalt (güvenlik için)
            if len(answer) > 600:
                answer = answer[:600] + "..."
            
            return answer
            
        except Exception as e:
            print(f"Chatbot hatası: {e}")
            return f"❌ Yanıt alınamadı: {str(e)[:50]}"
    
    def reset(self):
        """Sohbet geçmişini temizle"""
        if self.enabled and self.model:
            try:
                self.chat_session = self.model.start_chat(history=[])
                return "✅ Sohbet geçmişi temizlendi"
            except:
                return "❌ Sıfırlama başarısız"
        return "❌ Chatbot aktif değil"