# utils/timer.py
# FİNAL SÜRÜM (v2) - İLERLEME ÇUBUĞU İÇİN OPTİMİZE EDİLDİ

import time
from utils.logger import log_exercise

class DurationTimer:
    """
    Belirlenen bir süre boyunca hareketi takip eder.
    Sadece hareket algılandığında (is_active=True) süre işler.
    """
    
    def __init__(self, exercise_name, side, target_duration):
        self.exercise_name = exercise_name
        self.side = side
        self.target_duration = target_duration  # Hedef saniye
        self.reset()

    def reset(self):
        """Zamanlayıcıyı sıfırlar."""
        self.start_time = None
        self.elapsed_time = 0 # Toplam geçen süre
        self.last_update_time = None
        self.is_running = False
        self.is_complete = False
        self.logged = False
        self.last_message = f"{self.target_duration} sn Bekle"

    def update_feedback(self, is_active):
        """
        is_active: Kullanıcı doğru pozisyonda mı? (True/False)
        """
        
        if self.is_complete:
            return "TAMAMLANDI!"

        if is_active:
            current_time = time.time()
            if not self.is_running:
                # Sayım yeni başlıyor veya devam ediyor
                self.is_running = True
                self.last_update_time = current_time
            
            # Geçen süreyi ekle
            delta = current_time - self.last_update_time
            self.elapsed_time += delta
            self.last_update_time = current_time
            
            # Hedef kontrolü
            if self.elapsed_time >= self.target_duration:
                self.is_complete = True
                self.is_running = False
                if not self.logged:
                    log_exercise(self.exercise_name, self.target_duration, self.side)
                    self.logged = True
                return "TAMAMLANDI!"
            else:
                # Main.py regex'i için format: "X / Y"
                # Örn: "TUT! 3/10" -> Main.py bunu bar'a çevirir.
                remaining = int(self.target_duration - self.elapsed_time)
                elapsed_int = int(self.elapsed_time)
                self.last_message = f"TUT! {elapsed_int}/{self.target_duration}"
        
        else:
            # Pozisyon bozuldu, zamanı durdur ama sıfırlama (isteğe bağlı)
            # İstersen self.elapsed_time = 0 yaparak hatada sıfırlatabilirsin.
            # Şimdilik duraklatıyoruz:
            self.is_running = False
            self.last_update_time = None
            if self.elapsed_time > 0:
                self.last_message = f"DURDUN! {int(self.elapsed_time)}/{self.target_duration}"
            else:
                self.last_message = f"Pozisyon Al ({self.target_duration} sn)"
                
        return self.last_message