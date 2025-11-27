# utils/counter.py
# GÜNCELLENDİ (Mesaj sadeleştirildi, "Hedef Açı" kaldırıldı)

from utils.logger import log_exercise

class RepCounter:
    """
    ROM (Hareket Açıklığı) egzersizleri için tekrar sayar.
    """
    
    def __init__(self, exercise_name, side, threshold_angle, target_reps, neutral_threshold=5):
        self.exercise_name = exercise_name 
        self.side = side             
        self.rep_count = 0
        self.state = "down" 
        self.threshold = threshold_angle 
        self.neutral_threshold = neutral_threshold 
        self.target = target_reps 
        self.logged = False
        self.last_message = ""
        self.update_message() 

    def update_message(self, extra_text=""):
        """ Sadece mevcut duruma göre mesajı GÜNCELLER, saymaz. """
        if self.rep_count >= self.target:
            self.last_message = f"TAMAMLANDI! ({self.rep_count}/{self.target})"
        elif self.state == "up":
            self.last_message = f"{self.side}: {self.rep_count}/{self.target} (Merkeze don)"
        else: # state == "down"
            # "Pro" Düzeltme: Sadeleştirildi (Hedef kaldırıldı)
            self.last_message = f"{self.side}: {self.rep_count}/{self.target}"
        
        if extra_text:
            self.last_message = extra_text
            
        return self.last_message

    def get_current_message(self):
        """ Sadece son mesajı döndürür, durumu DEĞİŞTİRMEZ. """
        return self.last_message

    def count(self, current_angle):
        """ Açıya göre durumu GÜNCELLER ve yeni mesajı döndürür. """
        
        if self.rep_count >= self.target:
            if not self.logged:
                log_exercise(self.exercise_name, self.target, self.side)
                self.logged = True
            return self.update_message() 

        if self.state == "down":
            if current_angle > self.threshold:
                self.state = "up"
            return self.update_message() 
        
        elif self.state == "up":
            if current_angle < self.neutral_threshold: 
                self.rep_count += 1
                self.state = "down"
                
                if self.rep_count >= self.target:
                    return self.update_message()
                else:
                    return self.update_message(f"HARIKA! ({self.rep_count}/{self.target})")
            
            return self.update_message()
    
    def reset(self):
        """ Sayacı sıfırlar. """
        self.rep_count = 0
        self.state = "down"
        self.logged = False
        self.update_message()