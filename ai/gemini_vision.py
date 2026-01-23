# ai/gemini_vision.py
import google.generativeai as genai
import cv2
import base64
from PIL import Image
import io
import time

class GeminiVisionAnalyzer:
    """
    Gemini Vision ile egzersiz formu analizi
    """
    
    def __init__(self, api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.enabled = True
        else:
            self.enabled = False
            print("⚠️ Gemini API key bulunamadı. Vision özelliği devre dışı.")
        
        self.last_analysis_time = 0
        self.analysis_cooldown = 10  # 10 saniyede bir analiz
    
    def analyze_exercise_form(self, frame, exercise_name, current_angle=None):
        """
        Egzersiz formu analizi
        
        Args:
            frame: OpenCV frame (numpy array)
            exercise_name: Egzersiz kodu
            current_angle: Mevcut açı (opsiyonel)
        
        Returns:
            dict: {
                'feedback': str (geri bildirim),
                'quality_score': float (0-1),
                'suggestions': list (öneriler)
            }
        """
        if not self.enabled:
            return None
        
        # Cooldown kontrolü
        current_time = time.time()
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return None
        
        try:
            # Frame'i PIL Image'e çevir
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Prompt oluştur
            prompt = self._create_prompt(exercise_name, current_angle)
            
            # Gemini'ye gönder
            response = self.model.generate_content([prompt, pil_image])
            
            self.last_analysis_time = current_time
            
            # Yanıtı işle
            return self._parse_response(response.text)
        
        except Exception as e:
            print(f"Gemini Vision hatası: {e}")
            return None
    
    def _create_prompt(self, exercise_name, current_angle):
        """Egzersize özel prompt oluştur"""
        
        exercise_instructions = {
            "ROM_LAT": "Yana eğilme hareketi. Kulak omuza yaklaşmalı, omuzlar sabit kalmalı.",
            "ROM_ROT": "Boyun rotasyonu. Baş yana dönmeli, omuzlar sabit.",
            "OMUZ_YANA_ACMA": "Kollar yana kaldırılıyor. Omuzlar aşağıda, sırt düz.",
            "DIZ_HAVLU_EZME": "Diz altındaki havlu eziliyor. Bacak düz, topuk yerde.",
            "KALCA_KOPRU": "Köprü hareketi. Kalça havada, sırt düz, boyun nötr."
        }
        
        instruction = exercise_instructions.get(exercise_name, "Fizyoterapi egzersizi")
        
        prompt = f"""
Sen bir uzman fizyoterapistsin. Bu görüntüde hasta {exercise_name} egzersizini yapıyor.

EGZERSIZ: {instruction}

GÖREVIN:
1. Hastanın postür/formunu analiz et
2. Hataları tespit et (kompansasyon, yanlış hizalama)
3. 3 somut öneri ver

CEVAP FORMATI (Türkçe, kısa ve net):
FORM KALITESI: [Mükemmel/İyi/Orta/Zayıf]
HATALAR: [Varsa liste, yoksa "Yok"]
ÖNERİLER: [3 madde]

NOT: Çok kısa ve öz yaz, maksimum 100 kelime.
"""
        
        if current_angle:
            prompt += f"\n\nMEVCUT AÇI: {current_angle}°"
        
        return prompt
    
    def _parse_response(self, response_text):
        """Gemini yanıtını parse et"""
        
        # Basit parsing
        quality_map = {
            "Mükemmel": 1.0,
            "İyi": 0.8,
            "Orta": 0.6,
            "Zayıf": 0.4
        }
        
        quality_score = 0.7  # Varsayılan
        for word, score in quality_map.items():
            if word in response_text:
                quality_score = score
                break
        
        # Önerileri çıkar
        suggestions = []
        if "ÖNERİLER:" in response_text:
            suggestions_part = response_text.split("ÖNERİLER:")[1].strip()
            suggestions = [s.strip() for s in suggestions_part.split('\n') if s.strip()]
        
        return {
            'feedback': response_text,
            'quality_score': quality_score,
            'suggestions': suggestions[:3]  # İlk 3'ü al
        }
    
    def quick_check(self, frame, exercise_name):
        """Hızlı form kontrolü (sadece evet/hayır)"""
        if not self.enabled:
            return True
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            prompt = f"""
            Bu görüntüde {exercise_name} egzersizi yapılıyor.
            Form DOĞRU mu YANLIŞ mı? Tek kelime cevap: DOĞRU veya YANLIŞ
            """
            
            response = self.model.generate_content([prompt, pil_image])
            return "DOĞRU" in response.text.upper()
        
        except:
            return True  # Hata durumunda devam et


# ==================== KULLANIM ÖRNEĞİ ====================
if __name__ == "__main__":
    # API key gerekli (çevre değişkeni veya doğrudan)
    # export GEMINI_API_KEY="your-api-key"
    import os
    api_key = os.getenv("GEMINI_API_KEY")
    
    analyzer = GeminiVisionAnalyzer(api_key=api_key)
    
    if analyzer.enabled:
        # Test frame'i oku
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("Analiz yapılıyor...")
            result = analyzer.analyze_exercise_form(frame, "ROM_LAT", current_angle=35)
            
            if result:
                print("\n" + "="*50)
                print("GEMINI VISION ANALİZİ")
                print("="*50)
                print(f"Kalite Skoru: {result['quality_score']}")
                print(f"\n{result['feedback']}")
                print("="*50)
    else:
        print("Vision özelliği kullanılamıyor.")