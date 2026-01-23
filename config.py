# config.py
import os

class Config:
    """
    FizyoAsistan AI yapılandırma ayarları
    """
    
    # ==================== API KEYS ====================
    # Gemini API Key (https://makersuite.google.com/app/apikey)
    # ÖNEMLİ: Aşağıdaki satırı düzelt!
    GEMINI_API_KEY = "AIzaSyBQUzyGcWm9voPr9vvStpoiW37xBykZka0"  # Direkt string olarak yaz
    # VEYA environment variable kullan:
    # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
    
    # ==================== ÖZELLIK AÇMA/KAPATMA ====================
    VOICE_ENABLED = True           # Sesli asistan
    VISION_ENABLED = False         # Gemini Vision (API key gerekli)
    CHATBOT_ENABLED = False        # Chatbot (API key gerekli)
    ANALYTICS_ENABLED = True       # Grafikler ve analitik
    PLAN_ENABLED = True            # Kişisel plan oluşturma
    PAIN_PREDICTION_ENABLED = True # Ağrı tahmini
    
    # ==================== SES AYARLARI ====================
    VOICE_RATE = 160               # Konuşma hızı (100-200 arası)
    VOICE_VOLUME = 0.9             # Ses seviyesi (0.0-1.0)
    VOICE_COOLDOWN = 3             # Aynı mesajı X saniyede bir söyle
    
    # ==================== VISION AYARLARI ====================
    ANALYSIS_INTERVAL = 15         # Vision analizi aralığı (saniye)
    QUALITY_THRESHOLD = 0.5        # Form kalitesi eşiği (0-1)
    
    # ==================== GENEL AYARLAR ====================
    DATA_FOLDER = "data"           # Veri klasörü
    VIDEO_FOLDER = "videos"        # Video klasörü
    
    # Egzersiz limitleri
    MAX_REPS = 10
    DEFAULT_FITNESS_LEVEL = 5      # 1-10 arası
    DEFAULT_PLAN_WEEKS = 2
    
    # ==================== DEBUG ====================
    DEBUG_MODE = False             # Debug mesajları göster
    SHOW_FPS = False               # FPS göster
    
    @classmethod
    def check_requirements(cls):
        """Gereksinimleri kontrol et"""
        messages = []
        
        # API Key kontrolü
        if cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "BURAYA_API_KEY_GIR":
            messages.append("✅ Gemini API Key bulundu")
            cls.VISION_ENABLED = True
            cls.CHATBOT_ENABLED = True
        else:
            messages.append("⚠️ Gemini API Key yok - Vision ve Chatbot kapalı")
            cls.VISION_ENABLED = False
            cls.CHATBOT_ENABLED = False
        
        # Klasör kontrolü
        import os
        if not os.path.exists(cls.DATA_FOLDER):
            os.makedirs(cls.DATA_FOLDER)
            messages.append(f"✅ {cls.DATA_FOLDER} klasörü oluşturuldu")
        
        if not os.path.exists(cls.VIDEO_FOLDER):
            os.makedirs(cls.VIDEO_FOLDER)
            messages.append(f"✅ {cls.VIDEO_FOLDER} klasörü oluşturuldu")
        
        return messages


# Program başladığında gereksinimleri kontrol et
if __name__ != "__main__":  # Import edildiğinde çalıştır
    for msg in Config.check_requirements():
        print(msg)