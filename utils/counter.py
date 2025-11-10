# utils/counter.py
# GÜNCELLENDİ (Daha akıllı hale getirildi, "get_current_message" eklendi)

from utils.logger import log_exercise

class RepCounter:
    """
    ROM (Hareket Açıklığı) egzersizleri için tekrar sayar.
    Artık 'get_current_message' özelliği var, böylece spam yapmıyor.
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
        
        # YENİ: Son mesajı hafızada tut
        self.last_message = ""
        self.update_message() # İlk mesajı ayarla

    def update_message(self, extra_text=""):
        """ Sadece mevcut duruma göre mesajı GÜNCELLER, saymaz. """
        if self.rep_count >= self.target:
            self.last_message = f"TAMAMLANDI! ({self.rep_count}/{self.target})"
        elif self.state == "up":
            self.last_message = f"{self.side}: {self.rep_count}/{self.target} (Merkeze don)"
        else: # state == "down"
            self.last_message = f"{self.side}: {self.rep_count}/{self.target}"
        
        # Eğer "HARIKA!" gibi özel bir mesaj geldiyse
        if extra_text:
            self.last_message = extra_text
            
        return self.last_message

    def get_current_message(self):
        """ Sadece son mesajı döndürür, durumu DEĞİŞTİRMEZ. """
        return self.last_message

    def count(self, current_angle):
        """ Açıya göre durumu GÜNCELLER ve yeni mesajı döndürür. """
        
        # Eğer hedefi tamamladıysak
        if self.rep_count >= self.target:
            if not self.logged:
                log_exercise(self.exercise_name, self.target, self.side)
                self.logged = True
            return self.update_message() 

        # DURUM 1: NÖTR (DOWN)
        if self.state == "down":
            if current_angle > self.threshold:
                self.state = "up"
            return self.update_message() 
        
        # DURUM 2: HEDEFTE (UP)
        elif self.state == "up":
            # "PRO" HAREKET: Tam merkeze (örn: 2 derecenin altına) dönmeden sayma!
            if current_angle < self.neutral_threshold: 
                self.rep_count += 1
                self.state = "down"
                
                if self.rep_count >= self.target:
                    return self.update_message()
                else:
                    # Sadece bu anlık "HARIKA!" mesajını yolla
                    return self.update_message(f"HARIKA! ({self.rep_count}/{self.target})")
            
            # Henüz merkeze dönmedi
            return self.update_message()
    
    def reset(self):
        """ Sayacı sıfırlar. """
        self.rep_count = 0
        self.state = "down"
        self.logged = False
        self.update_message() # Mesajı sıfırla