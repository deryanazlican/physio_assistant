# main.py
# GÜNCELLENMİŞ TAM KOD (Diz Modülü + Omuz No. 14 Germe Dahil)

import cv2
import mediapipe as mp
import numpy as np

# --- Modül Importları ---
from utils.angles import calculate_angle_3d 
import exercises.boyun as boyun_modulu
import exercises.omuz as omuz_modulu  
import exercises.diz as diz_modulu # DİZ MODÜLÜ

# --- BUTON VE MENÜ AYARLARI ---
CURRENT_EXERCISE = "MENU_ANA" 
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

# --- FARE TIKLAMA FONKSİYONU (GÜNCELLENDİ) ---
def mouse_click_event(event, x, y, flags, param):
    global CURRENT_EXERCISE
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for button in BUTTON_LIST:
            if button.check_click(x, y):
                # Menülere girerken ilgili sayaçları sıfırla
                if button.exercise_name == "MENU_BOYUN":
                    boyun_modulu.reset_boyun_counters()
                elif button.exercise_name in ["MENU_OMUZ", "MENU_OMUZ_SOPA", "MENU_OMUZ_PEN", "MENU_OMUZ_DUVAR"]:
                    omuz_modulu.reset_omuz_counters() 
                elif button.exercise_name == "MENU_DIZ": 
                    diz_modulu.reset_diz_counters()
                    
                CURRENT_EXERCISE = button.exercise_name
                print(f"Secilen Egzersiz/Menu: {CURRENT_EXERCISE}")
                return

# --- Butonları Oluştur (TÜM MENÜLER) ---
b_width = 400
b_height = 55 
b_geri_pos = (1050, 600)
b_geri_size = (200, 80)

# 1. Ana Menü Butonları (GÜNCELLENDİ)
button_sec_boyun = Button((50, 325), b_width, b_height, "1. Boyun Egzersizleri", "MENU_BOYUN")
button_sec_omuz = Button((50, 390), b_width, b_height, "2. Omuz Egzersizleri", "MENU_OMUZ")
button_sec_diz = Button((50, 455), b_width, b_height, "3. Diz Egzersizleri", "MENU_DIZ") 
ANA_MENU_BUTTONS = [button_sec_boyun, button_sec_omuz, button_sec_diz]
button_geri_ana = Button(b_geri_pos, b_geri_size[0], b_geri_size[1], "<- ANA MENU", "MENU_ANA")

# 2. Boyun Menü Butonları (Kısaltıldı - Tamamı kodunuzda var)
BOYUN_MENU_BUTTONS = [
    Button((50, 260), b_width, b_height, "1. Yana Egilme (ROM)", "ROM_LAT"),
    Button((50, 325), b_width, b_height, "2. Donme (ROM)", "ROM_ROT"),
    Button((50, 390), b_width, b_height, "3. ROM (One/Arkaya)", "ROM_FLEKS"),
    Button((50, 455), b_width, b_height, "4. Izometrik (One)", "IZO_FLEKS"),
    Button((50, 520), b_width, b_height, "5. Izometrik (Arkaya)", "IZO_EKST"), 
    Button((50, 585), b_width, b_height, "6. Izometrik (Yana)", "IZO_LAT"),
    Button((50, 650), b_width, b_height, "7. Izometrik (Dondurme)", "IZO_ROT"),
    button_geri_ana
]

# 3. Omuz Ana Menü Butonları (GÜNCELLENDİ)
OMUZ_MENU_BUTTONS = [
    Button((50, 325), b_width, b_height, "A. Sopa Egzersizleri", "MENU_OMUZ_SOPA"),
    Button((50, 390), b_width, b_height, "B. Sallanma (Pendul) Egz.", "MENU_OMUZ_PEN"),
    Button((50, 455), b_width, b_height, "C. Duvar & Germe Egz.", "MENU_OMUZ_DUVAR"), # İsim güncellendi
    button_geri_ana
]

# 4. Omuz Alt Menü Butonları
button_geri_omuz = Button(b_geri_pos, b_geri_size[0], b_geri_size[1], "<- Omuz Menusu", "MENU_OMUZ")
OMUZ_SOPA_BUTTONS = [
    Button((50, 260), b_width, b_height, "2. Sopa ile Yana Acma", "OMUZ_YANA_ACMA"),
    Button((50, 325), b_width, b_height, "3. Sopa ile Disa Acma", "OMUZ_DISA_ACMA"),
    Button((50, 390), b_width, b_height, "4. Sopa ile One Acma", "OMUZ_ONE_ACMA"),
    Button((50, 455), b_width, b_height, "5. Sopa ile Arkaya Acma", "OMUZ_ARKAYA_ACMA"),
    Button((50, 520), b_width, b_height, "6. Sopa ile Ice Acma", "OMUZ_ICE_ACMA"),
    button_geri_omuz
]
OMUZ_PEN_BUTTONS = [
    Button((50, 325), b_width, b_height, "7. Kolu Onde Sallama (Rep)", "OMUZ_PEN_FLEKSIYON"),
    Button((50, 390), b_width, b_height, "8. Kolu Yanda Sallama (Rep)", "OMUZ_PEN_ABDUKSIYON"),
    Button((50, 455), b_width, b_height, "9. Cember Cizme (15sn)", "OMUZ_CEMBER"), 
    button_geri_omuz
]
# GÜNCELLENDİ (Buton 14 eklendi)
OMUZ_DUVAR_BUTTONS = [
    Button((50, 260), b_width, b_height, "10. Duvara Yana Acma", "OMUZ_DUVAR_YANA"),
    Button((50, 325), b_width, b_height, "11. Duvara One Itme", "OMUZ_DUVAR_ONE"),
    Button((50, 390), b_width, b_height, "12. Duvara Geriye Itme", "OMUZ_DUVAR_GERIYE"),
    Button((50, 455), b_width, b_height, "13. Duvara Disa Itme", "OMUZ_DUVAR_DISA"),
    Button((50, 520), b_width, b_height, "14. Germe (15sn)", "OMUZ_GERME"), # YENİ BUTON
    button_geri_omuz
]

# 5. Diz Menü Butonları (YENİ)
DIZ_MENU_BUTTONS = [
    Button((50, 260), b_width, b_height, "1. Diz Germe (Oturarak)", "DIZ_GERME"),
    Button((50, 325), b_width, b_height, "2. Topuk Kaydirma (Yatarak)", "DIZ_TOPUK_KAYDIR"),
    Button((50, 390), b_width, b_height, "3. Duz Bacak Kaldirma (Yatarak)", "DIZ_BACAK_KALDIR"),
    Button((50, 455), b_width, b_height, "4. Oturarak Bacak Uzatma", "DIZ_OTUR_UZAT"),
    Button((50, 520), b_width, b_height, "5. Duvar Squat (10sn)", "DIZ_DUVAR_SQUAT"),
    Button((50, 585), b_width, b_height, "6. Otur Kalk", "DIZ_OTUR_KALK"),
    button_geri_ana
]

# 6. Egzersiz Sırasındaki Geri Butonları
BOYUN_EXERCISE_BUTTONS = [Button(b_geri_pos, b_geri_size[0], b_geri_size[1], "<- GERI (Boyun)", "MENU_BOYUN")]
OMUZ_SOPA_EX_BUTTONS = [Button(b_geri_pos, b_geri_size[0], b_geri_size[1], "<- GERI (Sopa)", "MENU_OMUZ_SOPA")]
OMUZ_PEN_EX_BUTTONS = [Button(b_geri_pos, b_geri_size[0], b_geri_size[1], "<- GERI (Sallanma)", "MENU_OMUZ_PEN")]
OMUZ_DUVAR_EX_BUTTONS = [Button(b_geri_pos, b_geri_size[0], b_geri_size[1], "<- GERI (Duvar/Germe)", "MENU_OMUZ_DUVAR")] # İsim güncellendi
DIZ_EXERCISE_BUTTONS = [Button(b_geri_pos, b_geri_size[0], b_geri_size[1], "<- GERI (Diz)", "MENU_DIZ")] 


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
        
        feedback_talimat = "" 
        feedback_mesaj = ""   
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- ANA YÖNLENDİRİCİ (GÜNCELLENDİ) ---
            
            # 1. Ana Menü
            if CURRENT_EXERCISE == "MENU_ANA":
                feedback_talimat = "Lutfen calismak istediginiz bolgeyi secin"
                feedback_mesaj = "FizyoAsistan v2.0"
                BUTTON_LIST = ANA_MENU_BUTTONS
            
            # 2. Boyun Menüsü
            elif CURRENT_EXERCISE == "MENU_BOYUN":
                feedback_talimat = "Lutfen bir boyun egzersizi secin"
                BUTTON_LIST = BOYUN_MENU_BUTTONS
            
            # 3. Omuz Menüleri
            elif CURRENT_EXERCISE == "MENU_OMUZ":
                feedback_talimat, feedback_mesaj = omuz_modulu.get_exercise_feedback(CURRENT_EXERCISE, landmarks)
                BUTTON_LIST = OMUZ_MENU_BUTTONS
            elif CURRENT_EXERCISE == "MENU_OMUZ_SOPA":
                feedback_talimat, feedback_mesaj = omuz_modulu.get_exercise_feedback(CURRENT_EXERCISE, landmarks)
                BUTTON_LIST = OMUZ_SOPA_BUTTONS
            elif CURRENT_EXERCISE == "MENU_OMUZ_PEN":
                feedback_talimat, feedback_mesaj = omuz_modulu.get_exercise_feedback(CURRENT_EXERCISE, landmarks)
                BUTTON_LIST = OMUZ_PEN_BUTTONS
            elif CURRENT_EXERCISE == "MENU_OMUZ_DUVAR":
                feedback_talimat, feedback_mesaj = omuz_modulu.get_exercise_feedback(CURRENT_EXERCISE, landmarks)
                BUTTON_LIST = OMUZ_DUVAR_BUTTONS

            # 4. Diz Menüsü
            elif CURRENT_EXERCISE == "MENU_DIZ":
                feedback_talimat, feedback_mesaj = diz_modulu.get_exercise_feedback(CURRENT_EXERCISE, landmarks)
                BUTTON_LIST = DIZ_MENU_BUTTONS

            # 5. Egzersiz Çağırma
            
            # Boyun Egzersizleri
            elif CURRENT_EXERCISE.startswith("BOYUN_") or CURRENT_EXERCISE in ["ROM_LAT", "ROM_ROT", "ROM_FLEKS", "IZO_FLEKS", "IZO_EKST", "IZO_LAT", "IZO_ROT"]:
                feedback_talimat, feedback_mesaj = boyun_modulu.get_exercise_feedback(CURRENT_EXERCISE, landmarks)
                BUTTON_LIST = BOYUN_EXERCISE_BUTTONS
            
            # Omuz Egzersizleri (GÜNCELLENDİ)
            elif CURRENT_EXERCISE.startswith("OMUZ_"):
                feedback_talimat, feedback_mesaj = omuz_modulu.get_exercise_feedback(CURRENT_EXERCISE, landmarks)
                if CURRENT_EXERCISE.startswith("OMUZ_PEN_") or CURRENT_EXERCISE == "OMUZ_CEMBER":
                    BUTTON_LIST = OMUZ_PEN_EX_BUTTONS
                elif CURRENT_EXERCISE.startswith("OMUZ_DUVAR_") or CURRENT_EXERCISE == "OMUZ_GERME": # GÜNCELLENDİ (Germe eklendi)
                    BUTTON_LIST = OMUZ_DUVAR_EX_BUTTONS
                else: 
                    BUTTON_LIST = OMUZ_SOPA_EX_BUTTONS
            
            # Diz Egzersizleri
            elif CURRENT_EXERCISE.startswith("DIZ_"): 
                feedback_talimat, feedback_mesaj = diz_modulu.get_exercise_feedback(CURRENT_EXERCISE, landmarks)
                BUTTON_LIST = DIZ_EXERCISE_BUTTONS
            
        except Exception as e:
            feedback_talimat = "Kullanici algilanmadi"
            feedback_mesaj = "Menuye donuluyor..."
            CURRENT_EXERCISE = "MENU_ANA"
            BUTTON_LIST = ANA_MENU_BUTTONS 
        
        # İskeleti çizdir
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
        # Talimat ve Mesajları Yazdır
        cv2.putText(image, feedback_talimat, (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image, feedback_mesaj, (50, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
            
        # Butonları çizdir
        for button in BUTTON_LIST:
            button.draw(image)
            
        cv2.imshow(WINDOW_NAME, image) 

cap.release()
cv2.destroyAllWindows()