# main.py
# V46 - İSİM HATASI (MENU_WIDTH) GİDERİLDİ + API DÜZELTİLDİ + HAREKETLER GERİ GELDİ

import cv2
import mediapipe as mp
import numpy as np
import math
import threading, time, os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont 

# --- MEVCUT MODÜLLER ---
import utils.reports as reports 
import exercises.boyun as boyun_modulu
import exercises.omuz as omuz_modulu   
import exercises.diz as diz_modulu 
import exercises.kalca as kalca_modulu
import exercises.bel as bel_modulu 

from core.voice_assistant import VoiceAssistant
from core.plan_generator import PersonalizedPlanGenerator
from core.analytics import ProgressAnalytics
from ai.gemini_vision import GeminiVisionAnalyzer
from ai.chatbot import PhysioChatbot
from ai.pain_predictor import SimplePainPredictor
from config import Config

# ==============================================================================
# 🚨 KRİTİK DEĞİŞKENLER (EN BAŞTA TANIMLADIK Kİ HATA VERMESİN)
# ==============================================================================
MENU_WIDTH = 280
SABIT_API_KEY = "AIzaSyBQUzyGcWm9voPr9vvStpoiW37xBykZka0"

# --- MODÜLLERİ BAŞLAT ---
try:
    # Config'i zorla güncelle
    Config.GEMINI_API_KEY = SABIT_API_KEY 
    
    voice_assistant = VoiceAssistant(enabled=True)
    planner = PersonalizedPlanGenerator()
    analytics = ProgressAnalytics()
    vision_analyzer = GeminiVisionAnalyzer(api_key=SABIT_API_KEY)
    chatbot = PhysioChatbot(api_key=SABIT_API_KEY)
    pain_predictor = SimplePainPredictor()
except Exception as e:
    print(f"Modül Hatası: {e}")
    class Dummy: 
        def speak(self, t): pass
        def ask(self, q): return "Hata"
        def toggle(self): return False
    voice_assistant = Dummy(); chatbot = Dummy()

class DummyAsistan:
    def __init__(self): self.durum = "Ses Kapali"
    def konus(self, metin): pass 
asistan = DummyAsistan()

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = os.path.join(BASE_DIR, "videos")
VIDEO_MAP = {}

RENK_ACCENT_CYAN = (255, 255, 0)
RENK_HOVER_BG = (200, 200, 0)
RENK_NORMAL_BG = (60, 60, 60)
RENK_BEYAZ = (240, 240, 240)
RENK_YESIL_ONAY = (50, 205, 50)
RENK_KIRMIZI_GERI = (50, 50, 220)

PROGRAM_DURUMU = "ISIM_GIRIS" 
CURRENT_EXERCISE = "MENU_ANA" 
PREVIOUS_EXERCISE = ""
BUTTON_LIST = [] 
LAST_REP_COUNT = 0 
IS_TASK_COMPLETED = False 
HASTA_ISMI = ""  
RAPOR_VERISI = [] 
IS_SPLIT_MODE = False 
SESSION_ERRORS = [] 
AI_COMMENT = "" 
AI_LOADING = False 

# Gerekli Değişkenler
exercise_recorded = False
chatbot_active = False
last_vision_check = 0
exercise_start_time = None
exercise_quality_scores = []
CHATBOT_SORU = ""
CHATBOT_CEVAP = ""
chat_history = []
cap_video = None

# --- YARDIMCI FONKSİYONLAR ---
def put_text_tr(img, text, pos, font_size=32, color=(255,255,255), bold=False, centered=False):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Font Yükleme (Hatasız)
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except: font = ImageFont.load_default()
    
    if bold:
        try: font = ImageFont.truetype("arialbd.ttf", font_size)
        except: pass
    
    x, y = pos
    if centered:
        try: bbox = draw.textbbox((0,0), text, font=font); text_width = bbox[2] - bbox[0]
        except: text_width = len(text) * (font_size * 0.6)
        x = (img.shape[1] - text_width) // 2
    
    draw.text((x, y), text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_progress_bar(img, value, max_value=10, is_timer=False):
    h, w, c = img.shape; bar_w = 25; bar_h = 300
    x_start = w - 50; y_start = h // 2 - bar_h // 2; y_end = y_start + bar_h
    if max_value == 0: max_value = 1
    ratio = min(value / max_value, 1.0)
    cv2.rectangle(img, (x_start, y_start), (x_start+bar_w, y_end), (40,40,40), -1)
    fill_height = int(bar_h * ratio)
    fill_color = RENK_YESIL_ONAY if ratio >= 1.0 else ((0,255,255) if ratio > 0.7 else RENK_ACCENT_CYAN)
    cv2.rectangle(img, (x_start, y_end-fill_height), (x_start+bar_w, y_end), fill_color, -1)
    cv2.rectangle(img, (x_start, y_start), (x_start+bar_w, y_end), (150,150,150), 2)
    text = f"{int(value)}s" if is_timer else f"{int(value)}/{int(max_value)}"
    img = put_text_tr(img, text, (x_start-15, y_end+15), 22, RENK_BEYAZ, True)
    return img

def draw_angle_display(img, angle, label="Açı"):
    h, w, c = img.shape; box_w = 120; box_h = 70; x = w - box_w - 20; y = 20
    overlay = img.copy()
    cv2.rectangle(overlay, (x,y), (x+box_w, y+box_h), (30,30,30), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (x,y), (x+box_w, y+box_h), RENK_ACCENT_CYAN, 2)
    img = put_text_tr(img, label, (x+10, y+5), 16, RENK_BEYAZ)
    img = put_text_tr(img, f"{int(angle)}°", (x+20, y+30), 28, RENK_ACCENT_CYAN, True)
    return img

def get_exercise_video_path(exercise_code):
    base_name = VIDEO_MAP.get(exercise_code, exercise_code)
    for ext in [".mp4", ".MP4", ".avi", ".mkv", ".mov", ".MOV"]:
        if os.path.exists(os.path.join(VIDEO_FOLDER, base_name + ext)): return os.path.join(VIDEO_FOLDER, base_name + ext)
    return None

def draw_visual_protractor(img, p1, p2, p3):
    try:
        h, w = img.shape[:2]
        x1, y1 = int(p1.x*w), int(p1.y*h); x2, y2 = int(p2.x*w), int(p2.y*h); x3, y3 = int(p3.x*w), int(p3.y*h)
        cv2.line(img, (x1,y1), (x2,y2), RENK_BEYAZ, 3); cv2.line(img, (x3,y3), (x2,y2), RENK_BEYAZ, 3)
        cv2.circle(img, (x1,y1), 6, RENK_ACCENT_CYAN, -1); cv2.circle(img, (x2,y2), 8, (0,0,255), -1)
        cv2.circle(img, (x3,y3), 6, RENK_ACCENT_CYAN, -1)
    except: pass

class Button:
    def __init__(self, pos, width, height, text, exercise_name, is_back_button=False):
        self.pos = pos; self.width = width; self.height = height; self.text = text
        self.exercise_name = exercise_name; self.x, self.y = pos; self.is_back_button = is_back_button; self.is_hovered = False
    def draw(self, img):
        bg_color = (RENK_KIRMIZI_GERI if self.is_back_button else RENK_HOVER_BG) if self.is_hovered else RENK_NORMAL_BG
        if self.exercise_name == "CHATBOT_AC": bg_color = (50, 150, 50) 
        cv2.rectangle(img, self.pos, (self.x+self.width, self.y+self.height), bg_color, -1)
        if self.is_hovered and not self.is_back_button: cv2.rectangle(img, self.pos, (self.x+self.width, self.y+self.height), RENK_ACCENT_CYAN, 2)
        img[:] = put_text_tr(img, self.text, (self.x+10, self.y+10), 18, RENK_BEYAZ, True)
    def check_hover(self, x, y): self.is_hovered = (self.x < x < self.x+self.width and self.y < y < self.y+self.height)
    def check_click(self, x, y): return self.x < x < self.x+self.width and self.y < y < self.y+self.height

def create_sidebar_buttons(titles, codes, back_btn=None):
    btns = []
    start_y = 120; b_height = 45; b_margin = 10
    for i, (text, code) in enumerate(zip(titles, codes)):
        y_pos = start_y + i * (b_height + b_margin)
        btns.append(Button((20, y_pos), MENU_WIDTH, b_height, text, code))
    if back_btn: btns.append(back_btn)
    return btns

# --- BUTON LİSTELERİ ---
back_to_main = Button((20, 650), MENU_WIDTH, 50, "< ANA MENÜ", "MENU_ANA", True) 
back_to_omuz = Button((20, 650), MENU_WIDTH, 50, "< OMUZ MENÜSÜ", "MENU_OMUZ", True)

ANA_MENU_BUTTONS = create_sidebar_buttons(
    ["1. Boyun Egzersizleri", "2. Omuz Egzersizleri", "3. Diz Egzersizleri", "4. Kalça Egzersizleri", "5. Bel Egzersizleri"], 
    ["MENU_BOYUN", "MENU_OMUZ", "MENU_DIZ", "MENU_KALCA", "MENU_BEL"]
)
# CHATBOT BUTONU EN ALTTA
chatbot_btn = Button((20, 580), MENU_WIDTH, 50, "💬 AI ASİSTAN", "CHATBOT_AC")
ANA_MENU_BUTTONS.append(chatbot_btn)

BOYUN_MENU_BUTTONS = create_sidebar_buttons(["1. Yana Eğilme", "2. Dönme", "3. Öne/Arkaya Eğilme", "4. İzometrik (Öne)", "5. İzometrik (Arkaya)", "6. İzometrik (Yana)", "7. Çember Çizme"], ["ROM_LAT", "ROM_ROT", "ROM_FLEKS", "IZO_FLEKS", "IZO_EKST", "IZO_LAT", "ROM_CEMBER"], back_to_main)
OMUZ_MENU_BUTTONS = create_sidebar_buttons(["A. Sopa Egzersizleri", "B. Sallanma (Pendul)", "C. Duvar & Germe"], ["MENU_OMUZ_SOPA", "MENU_OMUZ_PEN", "MENU_OMUZ_DUVAR"], back_to_main)
DIZ_MENU_BUTTONS = create_sidebar_buttons(["1. Havlu Ezme (5sn)", "2. Yüzüstü Bükme", "3. Yan Yatarak", "4. Oturarak Uzat", "5. Duvar Squat"], ["DIZ_HAVLU_EZME", "DIZ_YUZUSTU_BUKME", "DIZ_YAN_KALDIR", "DIZ_OTUR_UZAT", "DIZ_DUVAR_SQUAT"], back_to_main)
KALCA_MENU_BUTTONS = create_sidebar_buttons(["1. Dizi Göğse Çekme", "2. Düz Bacak Kaldır", "3. Köprü Kurma", "4. Yan Yatarak Açma", "5. Yüzüstü Kaldır", "6. Yan Diz Çekme"], ["KALCA_DIZ_CEKME", "KALCA_DUZ_KALDIR", "KALCA_KOPRU", "KALCA_YAN_ACMA", "KALCA_YUZUSTU", "KALCA_YAN_DIZ_CEKME"], back_to_main)
BEL_MENU_BUTTONS = create_sidebar_buttons(["1. Tek Diz Çekme", "2. Çift Diz Çekme", "3. Yarım Mekik", "4. Düz Bacak (SLR)", "5. Köprü (5sn)", "6. Kedi - Deve", "7. Yüzüstü Doğrulma"], ["BEL_TEK_DIZ", "BEL_CIFT_DIZ", "BEL_MEKIK", "BEL_SLR", "BEL_KOPRU", "BEL_KEDI_DEVE", "BEL_YUZUSTU"], back_to_main)

OMUZ_SOPA_BUTTONS = create_sidebar_buttons(["1. Yana Açma", "2. Dışa Açma", "3. Öne Açma"], ["OMUZ_YANA_ACMA", "OMUZ_DISA_ACMA", "OMUZ_ONE_ACMA"], back_to_omuz)
OMUZ_PEN_BUTTONS = create_sidebar_buttons(["4. Önde Sallama", "5. Yanda Sallama", "6. Çember Çizme"], ["OMUZ_PEN_FLEKSIYON", "OMUZ_PEN_ABDUKSIYON", "OMUZ_CEMBER"], back_to_omuz)
OMUZ_DUVAR_BUTTONS = create_sidebar_buttons(["7. Duvara Yana", "8. Duvara Öne", "9. Duvara Geriye", "10. Germe (15sn)"], ["OMUZ_DUVAR_YANA", "OMUZ_DUVAR_ONE", "OMUZ_DUVAR_GERIYE", "OMUZ_GERME"], back_to_omuz)

def single_back(target): return [Button((20, 650), MENU_WIDTH, 50, "< GERI DON", target, True)]
BOYUN_EX_BTNS = single_back("MENU_BOYUN")
OMUZ_EX_BTNS = single_back("MENU_OMUZ") 
DIZ_EX_BTNS = single_back("MENU_DIZ")
KALCA_EX_BTNS = single_back("MENU_KALCA")
BEL_EX_BTNS = single_back("MENU_BEL")

# --- MOUSE OLAYLARI ---
def mouse_click_event(event, x, y, flags, param):
    global CURRENT_EXERCISE, PROGRAM_DURUMU, LAST_REP_COUNT, IS_TASK_COMPLETED, IS_SPLIT_MODE, SESSION_ERRORS, AI_COMMENT, chatbot_active, CHATBOT_SORU, chat_history
    
    if PROGRAM_DURUMU == "ISIM_GIRIS": return
    
    if PROGRAM_DURUMU == "GIRIS_EKRANI":
        if event == cv2.EVENT_LBUTTONDOWN:
            if 440 < x < 840 and 500 < y < 580:
                PROGRAM_DURUMU = "EGZERSIZ_MODU"
                voice_assistant.speak("Egzersiz modu.")
        return
    
    if PROGRAM_DURUMU == "AI_RAPOR_EKRANI":
        if event == cv2.EVENT_LBUTTONDOWN:
            if 440 < x < 840 and 600 < y < 660:
                PROGRAM_DURUMU = "EGZERSIZ_MODU"; SESSION_ERRORS = []; AI_COMMENT = ""; LAST_REP_COUNT = 0; IS_TASK_COMPLETED = False; voice_assistant.speak("Menüye dönüldü.")
        return

    adjusted_x = x
    if IS_SPLIT_MODE:
        if x < 640: return 
        adjusted_x = x - 640
    
    if event == cv2.EVENT_MOUSEMOVE:
        for button in BUTTON_LIST: button.check_hover(adjusted_x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for button in BUTTON_LIST:
            if button.check_click(adjusted_x, y):
                if button.exercise_name == "CHATBOT_AC":
                    chatbot_active = True
                    chat_history = []
                    CHATBOT_SORU = ""
                    voice_assistant.speak("Asistan dinliyor")
                    return

                # Navigasyon
                if button.exercise_name == "MENU_ANA": CURRENT_EXERCISE = "MENU_ANA"
                elif button.exercise_name == "MENU_BOYUN": CURRENT_EXERCISE = "MENU_BOYUN"
                elif button.exercise_name == "MENU_OMUZ": CURRENT_EXERCISE = "MENU_OMUZ"
                elif button.exercise_name == "MENU_DIZ": CURRENT_EXERCISE = "MENU_DIZ"
                elif button.exercise_name == "MENU_KALCA": CURRENT_EXERCISE = "MENU_KALCA"
                elif button.exercise_name == "MENU_BEL": CURRENT_EXERCISE = "MENU_BEL"
                elif button.exercise_name == "MENU_OMUZ_SOPA": CURRENT_EXERCISE = "MENU_OMUZ_SOPA"
                elif button.exercise_name == "MENU_OMUZ_PEN": CURRENT_EXERCISE = "MENU_OMUZ_PEN"
                elif button.exercise_name == "MENU_OMUZ_DUVAR": CURRENT_EXERCISE = "MENU_OMUZ_DUVAR"
                
                # Egzersiz Seçimi
                else:
                    if "BOYUN" in button.exercise_name or "ROM_" in button.exercise_name or "IZO_" in button.exercise_name: boyun_modulu.reset_boyun_counters()
                    elif "OMUZ" in button.exercise_name: omuz_modulu.reset_omuz_counters()
                    elif "DIZ" in button.exercise_name: diz_modulu.reset_diz_counters()
                    elif "KALCA" in button.exercise_name: kalca_modulu.reset_kalca_counters()
                    elif "BEL" in button.exercise_name: bel_modulu.reset_bel_counters()
                    
                    CURRENT_EXERCISE = button.exercise_name
                    LAST_REP_COUNT = 0; IS_TASK_COMPLETED = False; SESSION_ERRORS = []; AI_COMMENT = "" 
                    voice_assistant.speak(f"{button.text} seçildi")
                return

mp_drawing = mp.solutions.drawing_utils; mp_pose = mp.solutions.pose
cap_cam = cv2.VideoCapture(0); cap_cam.set(3, 1280); cap_cam.set(4, 720) 
WINDOW_NAME = 'FizyoAsistan AI v46'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL); cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(WINDOW_NAME, mouse_click_event)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        success_cam, frame_full = cap_cam.read()
        if not success_cam: continue
        frame_full = cv2.flip(frame_full, 1); frame_full = cv2.resize(frame_full, (1280, 720)) 
        key = cv2.waitKey(5) & 0xFF
        
        # --- CHATBOT PENCERESİ ---
        if chatbot_active:
            if key == 27: chatbot_active = False # ESC
            elif key == 13 and len(CHATBOT_SORU) > 2: # Enter
                chat_history.append(f"Sen: {CHATBOT_SORU}")
                try: 
                    cevap = chatbot.ask(CHATBOT_SORU)
                    chat_history.append(f"AI: {cevap}")
                except Exception as e: chat_history.append(f"Hata: {str(e)[:50]}")
                CHATBOT_SORU = ""
            elif key == 8: CHATBOT_SORU = CHATBOT_SORU[:-1]
            elif 32 <= key <= 126: CHATBOT_SORU += chr(key)
            
            overlay = frame_full.copy()
            cv2.rectangle(overlay, (200,100), (1080,620), (30,30,30), -1)
            frame_full = cv2.addWeighted(overlay, 0.95, frame_full, 0.05, 0)
            cv2.rectangle(frame_full, (200,100), (1080,620), RENK_ACCENT_CYAN, 2)
            frame_full = put_text_tr(frame_full, "AI ASISTAN", (0,120), 32, RENK_ACCENT_CYAN, True, centered=True)
            frame_full = put_text_tr(frame_full, "(ESC ile Kapat)", (0,160), 18, RENK_BEYAZ, False, centered=True)
            
            y = 220
            for msg in chat_history[-6:]:
                col = (100,255,100) if msg.startswith("Sen:") else (255,255,255)
                words = msg.split(" "); line = ""
                for w in words:
                    if len(line+w)<70: line += w + " "
                    else: frame_full=put_text_tr(frame_full,line,(220,y),20,col); y+=30; line=w+" "
                frame_full=put_text_tr(frame_full,line,(220,y),20,col); y+=40
            cv2.rectangle(frame_full, (220,550), (1060,600), RENK_ACCENT_CYAN, 2)
            frame_full = put_text_tr(frame_full, CHATBOT_SORU+"|", (230,565), 22, RENK_BEYAZ)
            cv2.imshow(WINDOW_NAME, frame_full); continue

        # --- NORMAL KISAYOLLAR ---
        if key == ord('q'): break
        elif key == ord('v'): voice_assistant.toggle()

        # İSİM GİRİŞ
        if PROGRAM_DURUMU == "ISIM_GIRIS":
            overlay = frame_full.copy(); cv2.rectangle(overlay, (0, 0), (1280, 720), (20, 20, 20), -1)
            frame_full = cv2.addWeighted(overlay, 0.9, frame_full, 0.1, 0)
            frame_full = put_text_tr(frame_full, "HASTA KAYIT SISTEMI", (0, 290), 32, RENK_BEYAZ, True, centered=True)
            frame_full = put_text_tr(frame_full, "Adiniz Soyadiniz:", (0, 350), 24, (200, 200, 200), centered=True)
            frame_full = put_text_tr(frame_full, HASTA_ISMI + "|", (0, 400), 40, RENK_ACCENT_CYAN, True, centered=True)
            if key != 255:
                if key == 13 and len(HASTA_ISMI) > 2: PROGRAM_DURUMU = "GIRIS_EKRANI"; voice_assistant.speak(f"Merhaba {HASTA_ISMI}")
                elif key == 8: HASTA_ISMI = HASTA_ISMI[:-1]
                elif key >= 32: HASTA_ISMI += chr(key).upper()
            cv2.imshow(WINDOW_NAME, frame_full); continue
        
        # GİRİŞ EKRANI
        if PROGRAM_DURUMU == "GIRIS_EKRANI":
            overlay = frame_full.copy(); cv2.rectangle(overlay, (0, 0), (1280, 720), (20, 20, 20), -1)
            frame_full = cv2.addWeighted(overlay, 0.8, frame_full, 0.2, 0)
            frame_full = put_text_tr(frame_full, "FIZYO ASISTAN", (0, 200), 60, RENK_ACCENT_CYAN, True, centered=True)
            frame_full = put_text_tr(frame_full, f"MERHABA, {HASTA_ISMI}", (0, 300), 30, RENK_BEYAZ, True, centered=True)
            cv2.rectangle(frame_full, (440, 500), (840, 580), RENK_ACCENT_CYAN, 2)
            frame_full = put_text_tr(frame_full, "BASLA", (0, 530), 30, RENK_BEYAZ, True, centered=True)
            cv2.imshow(WINDOW_NAME, frame_full); continue

        # AI RAPOR EKRANI
        if PROGRAM_DURUMU == "AI_RAPOR_EKRANI":
            bg = np.zeros((720, 1280, 3), dtype=np.uint8)
            put_text_tr(bg, "ANALIZ RAPORU", (0, 80), 45, RENK_ACCENT_CYAN, True, centered=True)
            if AI_LOADING: put_text_tr(bg, "Doktor analizi hazirlaniyor...", (0, 300), 30, (200,200,200), centered=True)
            else:
                y_text = 180
                if AI_COMMENT:
                    words = AI_COMMENT.split(" "); line = ""
                    for word in words:
                        if len(line + word) < 60: line += word + " "
                        else: put_text_tr(bg, line, (80, y_text), 28, RENK_BEYAZ); y_text += 45; line = word + " "
                    put_text_tr(bg, line, (80, y_text), 28, RENK_BEYAZ)
                cv2.rectangle(bg, (440, 600), (840, 660), (50, 150, 50), -1)
                put_text_tr(bg, "MENÜYE DÖN", (0, 615), 32, RENK_BEYAZ, True, centered=True)
            cv2.imshow(WINDOW_NAME, bg); continue

        # EGZERSİZ MODU
        IS_SPLIT_MODE = not CURRENT_EXERCISE.startswith("MENU_")
        if IS_SPLIT_MODE: frame_process = cv2.resize(frame_full, (640, 720))
        else: frame_process = frame_full.copy()
            
        image_rgb = cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB); image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        
        feedback_talimat = ""; feedback_mesaj = ""
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame_process, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark
            try:
                if "OMUZ" in CURRENT_EXERCISE: draw_visual_protractor(frame_process, lm[23], lm[11], lm[13]); draw_visual_protractor(frame_process, lm[24], lm[12], lm[14]) 
                elif "DIZ" in CURRENT_EXERCISE: draw_visual_protractor(frame_process, lm[23], lm[25], lm[27]); draw_visual_protractor(frame_process, lm[24], lm[26], lm[28]) 
                elif "KALCA" in CURRENT_EXERCISE or "BEL" in CURRENT_EXERCISE: draw_visual_protractor(frame_process, lm[11], lm[23], lm[25]); draw_visual_protractor(frame_process, lm[12], lm[24], lm[26])
                elif "BOYUN" in CURRENT_EXERCISE or "ROM_" in CURRENT_EXERCISE or "IZO_" in CURRENT_EXERCISE:
                    draw_visual_protractor(frame_process, lm[0], lm[11], lm[12])
            except: pass
        
        # MENÜ YÖNETİMİ
        if CURRENT_EXERCISE == "MENU_ANA": BUTTON_LIST = ANA_MENU_BUTTONS; feedback_talimat = "HOSGELDINIZ"
        elif CURRENT_EXERCISE == "MENU_BOYUN": BUTTON_LIST = BOYUN_MENU_BUTTONS
        elif CURRENT_EXERCISE == "MENU_OMUZ": BUTTON_LIST = OMUZ_MENU_BUTTONS
        elif CURRENT_EXERCISE == "MENU_OMUZ_SOPA": BUTTON_LIST = OMUZ_SOPA_BUTTONS
        elif CURRENT_EXERCISE == "MENU_OMUZ_PEN": BUTTON_LIST = OMUZ_PEN_BUTTONS
        elif CURRENT_EXERCISE == "MENU_OMUZ_DUVAR": BUTTON_LIST = OMUZ_DUVAR_BUTTONS
        elif CURRENT_EXERCISE == "MENU_DIZ": BUTTON_LIST = DIZ_MENU_BUTTONS
        elif CURRENT_EXERCISE == "MENU_KALCA": BUTTON_LIST = KALCA_MENU_BUTTONS
        elif CURRENT_EXERCISE == "MENU_BEL": BUTTON_LIST = BEL_MENU_BUTTONS
        else:
            # Egzersiz Aktif
            if "BOYUN" in CURRENT_EXERCISE or "ROM_" in CURRENT_EXERCISE or "IZO_" in CURRENT_EXERCISE: BUTTON_LIST = BOYUN_EX_BTNS
            elif "OMUZ" in CURRENT_EXERCISE: 
                if "PEN" in CURRENT_EXERCISE: BUTTON_LIST = OMUZ_PEN_EX_BTNS
                elif "DUVAR" in CURRENT_EXERCISE: BUTTON_LIST = OMUZ_DUVAR_EX_BTNS
                else: BUTTON_LIST = OMUZ_SOPA_EX_BTNS
            elif "DIZ" in CURRENT_EXERCISE: BUTTON_LIST = DIZ_EX_BTNS
            elif "KALCA" in CURRENT_EXERCISE: BUTTON_LIST = KALCA_EX_BTNS
            elif "BEL" in CURRENT_EXERCISE: BUTTON_LIST = BEL_EX_BTNS

            try:
                ekstra_bilgi = {}
                if "BOYUN" in CURRENT_EXERCISE or "ROM_" in CURRENT_EXERCISE or "IZO_" in CURRENT_EXERCISE:
                    feedback_talimat, feedback_mesaj, ekstra_bilgi = boyun_modulu.get_exercise_feedback(CURRENT_EXERCISE, lm)
                elif "OMUZ" in CURRENT_EXERCISE: feedback_talimat, feedback_mesaj = omuz_modulu.get_exercise_feedback(CURRENT_EXERCISE, lm)
                elif "DIZ" in CURRENT_EXERCISE: feedback_talimat, feedback_mesaj, ekstra_bilgi = diz_modulu.get_exercise_feedback(CURRENT_EXERCISE, lm)
                elif "KALCA" in CURRENT_EXERCISE: feedback_talimat, feedback_mesaj, ekstra_bilgi = kalca_modulu.get_exercise_feedback(CURRENT_EXERCISE, lm)
                elif "BEL" in CURRENT_EXERCISE: feedback_talimat, feedback_mesaj, ekstra_bilgi = bel_modulu.get_exercise_feedback(CURRENT_EXERCISE, lm)
                
                # Vision Check
                if vision_analyzer.enabled and (time.time() - last_vision_check > 25): 
                    vision_analyzer.analyze_exercise_form(frame_process, CURRENT_EXERCISE)
                    last_vision_check = time.time()

                if "angle" in ekstra_bilgi: frame_process = draw_angle_display(frame_process, ekstra_bilgi["angle"], "Aci")
                if "reps" in ekstra_bilgi and not ("BOYUN" in CURRENT_EXERCISE or "ROM_" in CURRENT_EXERCISE): 
                    frame_process = draw_progress_bar(frame_process, ekstra_bilgi["reps"], 10, False)
                if "timer" in ekstra_bilgi: frame_process = draw_progress_bar(frame_process, ekstra_bilgi["timer"], 10, True)

                if feedback_mesaj:
                    nums = re.findall(r'\d+', feedback_mesaj)
                    curr_val = int(nums[0]) if nums else 0
                    if curr_val > LAST_REP_COUNT: LAST_REP_COUNT = curr_val; voice_assistant.count_rep(curr_val, 10)
                    if (curr_val >= 10 or "Tamam" in feedback_mesaj) and not IS_TASK_COMPLETED:
                        IS_TASK_COMPLETED = True; PROGRAM_DURUMU = "AI_RAPOR_EKRANI"; AI_LOADING = True
                        if not exercise_recorded: analytics.record_exercise(HASTA_ISMI, CURRENT_EXERCISE, {'reps':LAST_REP_COUNT}); exercise_recorded = True
                        voice_assistant.speak("Egzersiz Bitti")
                        def call_ai():
                            global AI_COMMENT, AI_LOADING
                            yorum = ai_coach.doktor_yorumu_al(CURRENT_EXERCISE, LAST_REP_COUNT, SESSION_ERRORS)
                            AI_COMMENT = yorum; AI_LOADING = False; voice_assistant.speak(yorum)
                        threading.Thread(target=call_ai).start()
            except: pass

        if not IS_SPLIT_MODE:
            overlay = frame_process.copy(); cv2.rectangle(overlay, (0, 0), (MENU_WIDTH + 20, 720), (30, 30, 30), -1) 
            frame_process = cv2.addWeighted(overlay, 0.6, frame_process, 0.4, 0)
        
        frame_process = put_text_tr(frame_process, f"{HASTA_ISMI}", (0, 20), 18, RENK_ACCENT_CYAN, True, centered=True)
        frame_process = put_text_tr(frame_process, feedback_talimat, (0, 50), 24, RENK_ACCENT_CYAN, True, centered=True)
        frame_process = put_text_tr(frame_process, feedback_mesaj, (0, 90), 20, RENK_BEYAZ, False, centered=True)
        
        for button in BUTTON_LIST: button.draw(frame_process)

        if IS_SPLIT_MODE:
            if CURRENT_EXERCISE != PREVIOUS_EXERCISE:
                path = get_exercise_video_path(CURRENT_EXERCISE)
                cap_video = cv2.VideoCapture(path) if path else None
                PREVIOUS_EXERCISE = CURRENT_EXERCISE
                exercise_start_time = time.time(); exercise_recorded = False
            
            frame_video = np.zeros((720, 640, 3), dtype=np.uint8)
            if cap_video and cap_video.isOpened():
                ret_vid, v_img = cap_video.read()
                if not ret_vid: cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0); ret_vid, v_img = cap_video.read()
                if ret_vid: frame_video = cv2.resize(v_img, (640, 720))
            final_view = cv2.hconcat([frame_video, frame_process])
        else: final_view = frame_process

        cv2.imshow(WINDOW_NAME, final_view)

cap_cam.release()
if cap_video: cap_video.release()
cv2.destroyAllWindows()