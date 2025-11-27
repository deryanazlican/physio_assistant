# utils/timer.py
# YENİ DOSYA - Süre Bazlı Egzersizler için Zamanlayıcı

import time
from utils.logger import log_exercise

class DurationTimer:
    """
    Belirlenen bir süre boyunca hareketi takip eder.
    Sadece hareket algılandığında süre işler.
    """
    
    def __init__(self, exercise_name, side, target_duration):
        self.exercise_name = exercise_name
        self.side = side
        self.target_duration = target_duration  # Saniye cinsinden
        
        self.reset()

    def reset(self):
        """Zamanlayıcıyı sıfırlar."""
        self.start_time = None
        self.elapsed_time = 0
        self.is_running = False
        self.is_complete = False
        self.logged = False
        self.last_message = f"{self.side}: Basla ({self.target_duration} sn)"
        print(f"{self.exercise_name} ({self.side}) zamanlayici sifirlandi.")

    def update_feedback(self, is_active):
        """
        Kullanıcının aktif olup olmadığına göre zamanlayıcıyı günceller ve mesaj döndürür.
        """
        
        if self.is_complete:
            return self.last_message

        # 1. Kullanıcı aktif (hareket ediyor)
        if is_active:
            if not self.is_running:
                # Zamanlayıcıyı başlat/devam ettir
                self.start_time = time.time()
                self.is_running = True
            
            # Geçen toplam süreyi hesapla
            current_total_elapsed = self.elapsed_time + (time.time() - self.start_time)
            
            if current_total_elapsed >= self.target_duration:
                # Hedef tamamlandı
                self.is_complete = True
                self.is_running = False
                self.last_message = "TAMAMLANDI!"
                if not self.logged:
                    log_exercise(self.exercise_name, self.target_duration, self.side)
                    self.logged = True
            else:
                # Devam ediyor
                self.last_message = f"Devam... {int(current_total_elapsed)}/{self.target_duration} sn"
        
        # 2. Kullanıcı aktif değil (durdu)
        else:
            if self.is_running:
                # Zamanlayıcıyı duraklat
                self.elapsed_time += (time.time() - self.start_time)
                self.is_running = False
            
            if self.elapsed_time > 0:
                self.last_message = f"Durdun. {int(self.elapsed_time)}/{self.target_duration} sn"
            else:
                self.last_message = f"{self.side}: Harekete Basla"
                
        return self.last_message