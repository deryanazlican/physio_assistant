# main.py
# GÜNCEL TAM KOD (MENÜYE "7. IZO (DONDURME)" HAREKETİ EKLENDİ - SON!)

import cv2
import mediapipe as mp
import numpy as np

from utils.angles import calculate_angle_3d 
import exercises.boyun as boyun_modulu

# --- BUTON VE MENÜ AYARLARI ---
CURRENT_EXERCISE = "MENU" 
BUTTON_LIST = [] 

class Button:
    def __init__(self, pos, width, height, text, exercise_name):
        self.pos = pos; self.width = width; self.height = height
        self.text = text; self.exercise_name = exercise_name
        self.x, self.y = pos
    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.x + self.width, self.y + self.height), (100, 100, 100), cv2.FILLED)
        cv2.rectangle(img, self.pos, (self.x + self.width, self.y + self.height), (255, 255, 255), 3)
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0] 
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    def check_click(self, x, y):
        if self.x < x < self.x + self.width and self.y < y < self.y + self.height:
            return True
        return False

# --- FARE TIKLAMA FONKSİYONU ---
def mouse_click_event(event, x, y, flags, param):
    global CURRENT_EXERCISE
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for button in BUTTON_LIST:
            if button.check_click(x, y):
                if button.exercise_name == "MENU":
                    boyun_modulu.reset_boyun_counters()
                CURRENT_EXERCISE = button.exercise_name
                print(f"Secilen Egzersiz: {CURRENT_EXERCISE}")
                return

# --- Butonları Oluştur (7 HAREKETLİ DÜZEN) ---
b_width = 400
b_height = 55 # Butonları sığdırmak için biraz küçülttük

button_rom_lat = Button((50, 260), b_width, b_height, "1. Yana Egilme (ROM)", "ROM_LAT")
button_rom_rot = Button((50, 325), b_width, b_height, "2. Donme (ROM)", "ROM_ROT")
button_rom_fleks = Button((50, 390), b_width, b_height, "3. ROM (One/Arkaya)", "ROM_FLEKS")
button_izo_fleks = Button((50, 455), b_width, b_height, "4. Izometrik (One)", "IZO_FLEKS")
button_izo_ekst = Button((50, 520), b_width, b_height, "5. Izometrik (Arkaya)", "IZO_EKST") 
button_izo_lat = Button((50, 585), b_width, b_height, "6. Izometrik (Yana)", "IZO_LAT")
button_izo_rot = Button((50, 650), b_width, b_height, "7. Izometrik (Dondurme)", "IZO_ROT") # YENİ

button_geri = Button((1050, 600), 200, 80, "<- GERI", "MENU")

MENU_BUTTONS = [button_rom_lat, button_rom_rot, button_rom_fleks, 
                button_izo_fleks, button_izo_ekst, button_izo_lat, button_izo_rot]
EXERCISE_BUTTONS = [button_geri]

# --- Ana Program ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
print("Kamera başlatılıyor... (cikis icin 'q' basin)")

WINDOW_NAME = 'FizyoAsistan'
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_click_event)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        
        image = cv2.flip(image, 1) 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True 
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("Program sonlandırılıyor...")
            break
        
        feedback_mesaj = "" 
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- MENÜ MANTIĞI (GÜNCELLENDİ) ---
            if CURRENT_EXERCISE == "MENU":
                feedback_mesaj = "Lutfen bir hareket secin:"
                BUTTON_LIST = MENU_BUTTONS
            
            elif CURRENT_EXERCISE == "ROM_LAT":
                feedback_mesaj = boyun_modulu.check_boyun_lateral_fleksiyon(landmarks)
                BUTTON_LIST = EXERCISE_BUTTONS
                
            elif CURRENT_EXERCISE == "ROM_ROT":
                feedback_mesaj = boyun_modulu.check_boyun_rotasyon(landmarks)
                BUTTON_LIST = EXERCISE_BUTTONS
            
            elif CURRENT_EXERCISE == "ROM_FLEKS":
                feedback_mesaj = boyun_modulu.check_boyun_fleksiyon_ekstansiyon(landmarks)
                BUTTON_LIST = EXERCISE_BUTTONS
            
            elif CURRENT_EXERCISE == "IZO_FLEKS":
                feedback_mesaj = boyun_modulu.check_boyun_izometrik_fleksiyon(landmarks)
                BUTTON_LIST = EXERCISE_BUTTONS
            
            elif CURRENT_EXERCISE == "IZO_EKST": 
                feedback_mesaj = boyun_modulu.check_boyun_izometrik_ekstansiyon(landmarks)
                BUTTON_LIST = EXERCISE_BUTTONS
                
            elif CURRENT_EXERCISE == "IZO_LAT":
                feedback_mesaj = boyun_modulu.check_boyun_izometrik_lateral(landmarks)
                BUTTON_LIST = EXERCISE_BUTTONS
                
            elif CURRENT_EXERCISE == "IZO_ROT": # YENİ
                feedback_mesaj = boyun_modulu.check_boyun_izometrik_rotasyon(landmarks)
                BUTTON_LIST = EXERCISE_BUTTONS
            
        except Exception as e:
            feedback_mesaj = "Kullanici algilanmadi"
            BUTTON_LIST = MENU_BUTTONS 
        
        # İskeleti çizdir
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
        # Geri bildirim mesajını yazdır
        cv2.putText(image, feedback_mesaj, (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            
        # Butonları çizdir
        for button in BUTTON_LIST:
            button.draw(image)
            
        cv2.imshow(WINDOW_NAME, image) 

cap.release()
cv2.destroyAllWindows()