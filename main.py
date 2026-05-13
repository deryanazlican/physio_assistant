# main.py
# V48.0 - AI CHAT PANEL + PLAN + SESSION + RAPOR + PDF + BUTTON SCROLL + PAIN PREDICTOR

from __future__ import annotations

import re
import cv2
from core.pose_backends.mediapipe_backend import MediaPipePoseBackend
from core.pose_backends.yolo_pose_backend import YOLOPoseBackend
# from core.pose_backends.movenet_backend import MoveNetBackend
import numpy as np
import time
import os
import json
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont

import utils.reports as reports
import exercises.boyun as boyun_modulu
import exercises.omuz as omuz_modulu
import exercises.diz as diz_modulu
import exercises.kalca as kalca_modulu
import exercises.bel as bel_modulu

from core.voice_assistant import VoiceAssistant
from core.plan_generator import PersonalizedPlanGenerator
from core.analytics import ProgressAnalytics
from core.session_manager import SessionManager
from core.personalization import build_patient_profile, generate_adaptive_recommendation
from core.progress_report import compare_progress_summaries

from core.patient_feedback import build_patient_feedback
from core.progress_report import compare_progress_summaries

from ai.chatbot import PhysioChatbot
from ai.pain_predictor import SimplePainPredictor
from ai.ml_pain_predictor import MLPainPredictor

from utils.chat_coordinator import ChatCoordinator
from utils.chat_panel import draw_chat_panel
from utils.reports import kaydet_session_raporu
from utils.pdf_report import export_session_pdf_auto
from utils.progress_dashboard import generate_progress_dashboard

from core.fps_counter import FPSCounter
from core.experiment_logger import ExperimentLogger

def draw_multiline_text_centered(
    frame,
    text,
    start_y,
    max_width,
    font_size=20,
    color=(255, 255, 255),
    line_spacing=30
):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = (current_line + " " + word).strip()

        # yaklaşık genişlik hesabı
        temp = np.zeros((100, 1200, 3), dtype=np.uint8)
        temp = put_text_tr(temp, test_line, (0, 0), font_size, color, False, centered=False)
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero((gray > 0).astype(np.uint8))
        width = 0
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            width = w

        if width > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    y = start_y
    for line in lines:
        frame = put_text_tr(
            frame,
            line,
            (0, y),
            font_size,
            color,
            False,
            centered=True
        )
        y += line_spacing

    return frame

def draw_multiline_text_left(
    frame,
    text,
    x,
    start_y,
    max_width,
    font_size=20,
    color=(255, 255, 255),
    line_spacing=28
):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = (current_line + " " + word).strip()

        temp = np.zeros((120, 1200, 3), dtype=np.uint8)
        temp = put_text_tr(temp, test_line, (0, 0), font_size, color, False, centered=False)
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero((gray > 0).astype(np.uint8))

        width = 0
        if coords is not None:
            xx, yy, ww, hh = cv2.boundingRect(coords)
            width = ww

        if width > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    y = start_y
    for line in lines:
        frame = put_text_tr(frame, line, (x, y), font_size, color, False, centered=False)
        y += line_spacing

    return frame

def draw_left_video_overlay(
    img,
    patient_name,
    pain_level,
    fps_value,
    model_name,
    angle_text=None,
    reps_text=None
):
    x = 18
    y = 18
    box_w = 235
    box_h = 34
    gap = 8

    items = [
        f"Ağrı: {pain_level}/10",
        f"FPS: {fps_value:.1f}",
        f"Model: {model_name}",
    ]

    if angle_text:
        items.append(angle_text)

    if reps_text:
        items.append(reps_text)

    for i, text in enumerate(items):
        y1 = y + i * (box_h + gap)
        y2 = y1 + box_h

        overlay = img.copy()
        cv2.rectangle(overlay, (x, y1), (x + box_w, y2), (35, 35, 35), -1)
        cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
        cv2.rectangle(img, (x, y1), (x + box_w, y2), RENK_ACCENT_CYAN, 2)

        img = put_text_tr(
            img,
            text,
            (x + 10, y1 + 7),
            15,
            RENK_BEYAZ,
            True
        )

    return img
# ==============================================================================
# SABİTLER
# ==============================================================================
MENU_WIDTH = 280
WINDOW_NAME = "FizyoAsistan v48.0"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = os.path.join(BASE_DIR, "videos")
HISTORY_FILE = os.path.join(BASE_DIR, "exercise_history.json")

MODEL_DIR = os.path.join(BASE_DIR, "models")
ML_MODEL_PATH = os.path.join(MODEL_DIR, "pain_predictor_ml.joblib")

VIDEO_MAP = {
    # İstersen eşleştirmeleri buraya gir
    # "ROM_LAT": "boyun_yana",
    # "ROM_ROT": "boyun_donme",
}

RENK_ACCENT_CYAN = (255, 255, 0)
RENK_HOVER_BG = (200, 200, 0)
RENK_NORMAL_BG = (60, 60, 60)
RENK_BEYAZ = (240, 240, 240)
RENK_YESIL_ONAY = (50, 205, 50)
RENK_KIRMIZI_GERI = (50, 50, 220)
RENK_PANEL = (35, 35, 35)

# ==============================================================================
# MODÜLLERİ BAŞLAT
# ==============================================================================
try:
    voice_assistant = VoiceAssistant(enabled=True)
    planner = PersonalizedPlanGenerator()
    analytics = ProgressAnalytics()
    session_manager = SessionManager()
    chatbot = PhysioChatbot(api_key=GEMINI_API_KEY)

    rule_based_predictor = SimplePainPredictor()
    ml_predictor = None

    if os.path.exists(ML_MODEL_PATH):
        temp_ml_predictor = MLPainPredictor(ML_MODEL_PATH)

        if temp_ml_predictor.is_loaded:
            ml_predictor = temp_ml_predictor
            pain_predictor = ml_predictor
            print("ML pain predictor aktif.")
        else:
            pain_predictor = rule_based_predictor
            print("ML model bulundu ama yüklenemedi. Rule-based predictor kullanılacak.")
    else:
        pain_predictor = rule_based_predictor
        print("ML model bulunamadı. Rule-based predictor kullanılacak.")

    print("pain_predictor type:", type(pain_predictor).__name__ if pain_predictor else None)
    print("rule_based_predictor type:", type(rule_based_predictor).__name__ if rule_based_predictor else None)
    print("ml_predictor type:", type(ml_predictor).__name__ if ml_predictor else None)

except Exception as e:
    print(f"Modül Hatası: {e}")

    class Dummy:
        def speak(self, t, priority=False):
            pass

        def ask(self, q):
            return "Hata"

        def toggle(self):
            return False

        def count_rep(self, count, total):
            pass

    voice_assistant = Dummy()
    chatbot = Dummy()
    session_manager = None
    planner = None
    analytics = None

    rule_based_predictor = SimplePainPredictor()
    ml_predictor = None
    pain_predictor = rule_based_predictor

# ==============================================================================
# GLOBAL DURUM
# ==============================================================================
PROGRAM_DURUMU = "ISIM_GIRIS"
CURRENT_EXERCISE = "MENU_ANA"
PREVIOUS_EXERCISE = ""
BUTTON_LIST = []
LAST_REP_COUNT = 0
IS_TASK_COMPLETED = False
HASTA_ISMI = ""
IS_SPLIT_MODE = False
SESSION_ERRORS: List[str] = []

PREDICTOR_MODE = "rule_based"

# Chatbot UI
chatbot_active = False
CHATBOT_SORU = ""
chat_scroll_offset = 0
chat_max_scroll = 0
chat_ui_info = {}
chat_controller = None

# Video
cap_video = None
exercise_start_time: Optional[float] = None
exercise_recorded = False

# Pain predictor / session analytics
last_prediction: Optional[Dict[str, Any]] = None
last_prediction_text: str = ""
last_exercise_timestamp: Optional[float] = None
current_pain_level: int = 3
exercise_history: List[Dict[str, Any]] = []

POST_EXERCISE_PAIN = 3
PENDING_EXERCISE_DATA = None


last_rule_prediction = None
last_ml_prediction = None

# Experiment / thesis metrics
fps_counter = FPSCounter(window_size=30)
experiment_logger = ExperimentLogger(os.path.join(BASE_DIR, "experiment_logs"))
CURRENT_MODEL_NAME = "mediapipe"
FRAME_INDEX = 0
CURRENT_FPS = 0.0
CURRENT_SESSION_LOG_PATH = None
LAST_PROGRESS_SUMMARY = None
LAST_PROGRESS_REPORT = None

LAST_PATIENT_FEEDBACK = ""
LAST_SPOKEN_GUIDANCE = ""
LAST_SPOKEN_GUIDANCE_TIME = 0.0

# ==============================================================================
# HISTORY YÖNETİMİ
# ==============================================================================
def load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"History okunamadı: {e}")
        return []


def save_history(history: List[Dict[str, Any]]) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"History kaydedilemedi: {e}")

def find_previous_session_summary(
    logs_dir: str,
    patient_name: str,
    exercise_code: str,
    current_log_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    def slugify_patient_name(name: str) -> str:
        import re
        name = (name or "UNKNOWN").strip().upper()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^A-Z0-9_ÇĞİÖŞÜ]", "", name)
        return name or "UNKNOWN"

    patient_file = os.path.join(logs_dir, f"{slugify_patient_name(patient_name)}.json")

    if not os.path.exists(patient_file):
        return None

    try:
        with open(patient_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        sessions = data.get("sessions", [])
        if not isinstance(sessions, list):
            return None

        exercise_code_norm = (exercise_code or "").strip().upper()

        matched = []
        for sess in sessions:
            sess_ex = str(sess.get("exercise_code", "")).strip().upper()
            if sess_ex != exercise_code_norm:
                continue

            summary = sess.get("summary", {}) or {}
            created_at = sess.get("created_at", "")
            matched.append((created_at, summary))

        if len(matched) < 2:
            return None

        matched.sort(key=lambda x: x[0])
        return matched[-2][1]

    except Exception as e:
        print(f"Patient log okuma hatası: {patient_file} -> {e}")
        return None
    
exercise_history = load_history()

# ==============================================================================
# YARDIMCI ÇİZİM FONKSİYONLARI
# ==============================================================================
def put_text_tr(img, text, pos, font_size=32, color=(255, 255, 255), bold=False, centered=False):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    if bold:
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except Exception:
            pass

    x, y = pos
    if centered:
        try:
            bbox = draw.textbbox((0, 0), str(text), font=font)
            text_width = bbox[2] - bbox[0]
        except Exception:
            text_width = int(len(str(text)) * (font_size * 0.6))
        x = (img.shape[1] - text_width) // 2

    draw.text((x, y), str(text), font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_progress_bar(img, value, max_value=10, is_timer=False):
    h, w, _ = img.shape
    bar_w = 25
    bar_h = 300
    x_start = w - 50
    y_start = h // 2 - bar_h // 2
    y_end = y_start + bar_h

    if max_value == 0:
        max_value = 1

    ratio = min(value / max_value, 1.0)
    cv2.rectangle(img, (x_start, y_start), (x_start + bar_w, y_end), (40, 40, 40), -1)

    fill_height = int(bar_h * ratio)
    fill_color = RENK_YESIL_ONAY if ratio >= 1.0 else ((0, 255, 255) if ratio > 0.7 else RENK_ACCENT_CYAN)

    cv2.rectangle(img, (x_start, y_end - fill_height), (x_start + bar_w, y_end), fill_color, -1)
    cv2.rectangle(img, (x_start, y_start), (x_start + bar_w, y_end), (150, 150, 150), 2)

    text = f"{int(value)}s" if is_timer else f"{int(value)}/{int(max_value)}"
    img = put_text_tr(img, text, (x_start - 15, y_end + 15), 22, RENK_BEYAZ, True)
    return img


def draw_angle_display(img, angle, label="Açı"):
    h, w, _ = img.shape
    box_w = 120
    box_h = 70
    x = w - box_w - 20
    y = 20

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), RENK_ACCENT_CYAN, 2)
    img = put_text_tr(img, label, (x + 10, y + 5), 16, RENK_BEYAZ)
    img = put_text_tr(img, f"{int(angle)}°", (x + 20, y + 30), 28, RENK_ACCENT_CYAN, True)
    return img


def get_exercise_video_path(exercise_code):
    base_name = VIDEO_MAP.get(exercise_code, exercise_code)
    for ext in [".mp4", ".MP4", ".avi", ".mkv", ".mov", ".MOV"]:
        p = os.path.join(VIDEO_FOLDER, base_name + ext)
        if os.path.exists(p):
            return p
    return None


def draw_visual_protractor(img, p1, p2, p3):
    try:
        h, w = img.shape[:2]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        x3, y3 = int(p3.x * w), int(p3.y * h)

        cv2.line(img, (x1, y1), (x2, y2), RENK_BEYAZ, 3)
        cv2.line(img, (x3, y3), (x2, y2), RENK_BEYAZ, 3)
        cv2.circle(img, (x1, y1), 6, RENK_ACCENT_CYAN, -1)
        cv2.circle(img, (x2, y2), 8, (0, 0, 255), -1)
        cv2.circle(img, (x3, y3), 6, RENK_ACCENT_CYAN, -1)
    except Exception:
        pass

def draw_pain_prediction_panel(img, prediction: Optional[Dict[str, Any]]):
    global last_rule_prediction, last_ml_prediction

    if not prediction:
        return img

    h, w = img.shape[:2]
    panel_w = 360
    panel_h = 250
    x1 = w - panel_w - 20
    y1 = h - panel_h - 20
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), RENK_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), RENK_ACCENT_CYAN, 2)

    risk_color = prediction.get("risk_color", "")
    risk_level = prediction.get("risk_level", "-")
    predicted_pain = prediction.get("predicted_pain", "-")
    warnings = prediction.get("warnings", []) or []
    recommendations = prediction.get("recommendations", []) or []

    img = put_text_tr(img, "Ağrı Risk Analizi", (x1 + 15, y1 + 10), 22, RENK_ACCENT_CYAN, True)
    img = put_text_tr(img, f"Risk: {risk_color} {risk_level}", (x1 + 15, y1 + 45), 18, RENK_BEYAZ, True)
    img = put_text_tr(img, f"Tahmini Ağrı: {predicted_pain}/10", (x1 + 15, y1 + 75), 18, RENK_BEYAZ, True)


    y_cursor = y1 + 110

    img = put_text_tr(img, "Uyarılar:", (x1 + 15, y_cursor), 16, (255, 220, 180), True)
    y_cursor += 24

    if warnings:
        cleaned = str(warnings[0]).replace("⚠️", "").strip()
        img = draw_multiline_text_left(
            img,
            f"- {cleaned}",
            x=x1 + 18,
            start_y=y_cursor,
            max_width=panel_w - 36,
            font_size=14,
            color=RENK_BEYAZ,
            line_spacing=18
        )
        y_cursor += 36
    else:
        img = put_text_tr(img, "- Uyarı yok.", (x1 + 18, y_cursor), 14, RENK_YESIL_ONAY)
        y_cursor += 20

    y_cursor += 8
    if recommendations:
        rec = str(recommendations[0]).replace("✅", "").strip()
        img = put_text_tr(img, "Öneri:", (x1 + 15, y_cursor), 16, RENK_ACCENT_CYAN, True)
        y_cursor += 22
        img = draw_multiline_text_left(
            img,
            rec,
            x=x1 + 18,
            start_y=y_cursor,
            max_width=panel_w - 36,
            font_size=14,
            color=RENK_ACCENT_CYAN,
            line_spacing=18
        )

    if last_rule_prediction is not None:
        rule_val = last_rule_prediction.get("predicted_pain", "-")
        rule_text = f"Rule: {rule_val}"

    if last_ml_prediction is not None:
        ml_val = last_ml_prediction.get("predicted_pain", "-")
        ml_text = f"ML: {ml_val}"


    return img

# ==============================================================================
# PAIN PREDICTOR HELPER'LARI
# ==============================================================================
def get_last_exercise_hours_ago() -> float:
    global last_exercise_timestamp
    if last_exercise_timestamp is None:
        return 48.0
    return max(0.0, (time.time() - last_exercise_timestamp) / 3600.0)


def calculate_quality_score(exercise_name: str, ekstra_bilgi: Optional[Dict[str, Any]], feedback_mesaj: str) -> float:
    """
    0-1 arasında quality skoru üretir.
    İlk sürüm: genel ve güvenli heuristics.
    """
    kalite = 1.0
    ekstra_bilgi = ekstra_bilgi or {}
    msg = (feedback_mesaj or "").lower()

    angle = float(ekstra_bilgi.get("angle", 0) or 0)
    reps = int(ekstra_bilgi.get("reps", 0) or 0)

    bad_keywords = [
        "hatalı",
        "yanlış",
        "dikkat",
        "yavaş",
        "omuz",
        "denge",
        "tam değil",
        "eksik",
        "stabil",
        "sabitle",
    ]

    for kw in bad_keywords:
        if kw in msg:
            kalite -= 0.08

    if "boyun" in exercise_name or "rom_" in exercise_name or "izo_" in exercise_name:
        if angle and angle < 15:
            kalite -= 0.20
        if "omuz" in msg:
            kalite -= 0.15
        if "çok hızlı" in msg or "hızlı" in msg:
            kalite -= 0.12

    elif "omuz" in exercise_name.lower():
        if angle and angle < 45:
            kalite -= 0.18
        if "gövde" in msg or "sallan" in msg:
            kalite -= 0.12

    elif "diz" in exercise_name.lower():
        if angle and angle < 40:
            kalite -= 0.18
        if "içe" in msg or "denge" in msg:
            kalite -= 0.12

    elif "kalca" in exercise_name.lower() or "bel" in exercise_name.lower():
        if angle and angle < 20:
            kalite -= 0.15
        if "beli sabit" in msg or "gövdeyi sabit" in msg:
            kalite -= 0.10

    if reps >= 10 and "iyi" in msg:
        kalite += 0.05

    kalite = max(0.0, min(1.0, kalite))
    return round(kalite, 2)

def run_pain_prediction(
    exercise_name: str,
    reps: int,
    duration_minutes: float,
    quality: float,
    current_pain: int,
    last_exercise_hours_ago: float,
    history: List[Dict[str, Any]],
    angle: float = 0.0,
) -> Dict[str, Any]:
    global last_prediction, pain_predictor
    global rule_based_predictor, ml_predictor
    global last_rule_prediction, last_ml_prediction

    current_data = {
        "exercise": exercise_name,
        "reps": reps,
        "duration": duration_minutes,
        "quality": quality,
        "current_pain": current_pain,
        "last_exercise_hours_ago": last_exercise_hours_ago,
        "angle": angle,
    }

    if rule_based_predictor is not None:
        try:
            last_rule_prediction = rule_based_predictor.predict_pain_after_exercise(current_data, history)
        except Exception as e:
            print(f"Rule-based prediction hatası: {e}")

    if ml_predictor is not None:
        try:
            last_ml_prediction = ml_predictor.predict_pain_after_exercise(current_data, history)
        except Exception as e:
            print(f"ML prediction hatası: {e}")

    use_ml = False

    if ml_predictor is not None and len(history) >= 20:
        use_ml = True

    if use_ml:
        prediction = ml_predictor.predict_pain_after_exercise(current_data, history)
    else:
        prediction = rule_based_predictor.predict_pain_after_exercise(current_data, history)

    last_prediction = prediction
    print("ACTIVE PREDICTION:", prediction)
    print("RULE PREDICTION:", last_rule_prediction)
    print("ML PREDICTION:", last_ml_prediction)
    return prediction

def finalize_exercise_and_predict(
    exercise_name: str,
    reps: int,
    duration_sec: float,
    quality: float,
    current_pain: int,
    angle: float = 0.0,
) -> Dict[str, Any]:
    global last_prediction_text, last_exercise_timestamp, HASTA_ISMI

    last_hours = get_last_exercise_hours_ago()

    prediction = run_pain_prediction(
        exercise_name=exercise_name,
        reps=reps,
        duration_minutes=max(0.0, duration_sec / 60.0),
        quality=quality,
        current_pain=current_pain,
        last_exercise_hours_ago=last_hours,
        history=exercise_history,
        angle=angle,
    )

    recommendation_text = pain_predictor.get_recommendation_text(prediction)
    last_prediction_text = recommendation_text

    record = {
        "patient_name": HASTA_ISMI,
        "exercise": exercise_name,
        "timestamp": time.time(),
        "data": {
            "angle": float(angle or 0.0),
            "quality": float(quality),
            "reps": int(reps),
            "duration": round(duration_sec / 60.0, 2),
            "current_pain": int(current_pain),
            "predicted_pain": float(prediction.get("predicted_pain", 0)),
            "risk_level": prediction.get("risk_level", ""),
            "last_exercise_hours_ago": float(last_hours),
            "pain_after": None
        },
    }

    last_exercise_timestamp = time.time()

    return {
        "prediction": prediction,
        "record": record
    }

def get_target_angle_for_exercise(exercise_code: str) -> Optional[float]:
    """
    Tez deneyleri için başlangıç hedef açıları.
    Sonra egzersize göre daha detaylı özelleştirebiliriz.
    """
    if "ROM_ROT" in exercise_code:
        return 30.0
    if "ROM_LAT" in exercise_code:
        return 20.0
    if "ROM_FLEKS" in exercise_code:
        return 25.0
    if "IZO_" in exercise_code:
        return 10.0
    if "OMUZ" in exercise_code:
        return 60.0
    if "DIZ" in exercise_code:
        return 45.0
    if "KALCA" in exercise_code:
        return 30.0
    if "BEL" in exercise_code:
        return 25.0
    return None

def update_experiment_log_pain_after(log_path: Optional[str], pain_after: int) -> None:
    if not log_path or not os.path.exists(log_path):
        print(f"[WARN] Log dosyası bulunamadı: {log_path}")
        return

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        summary["pain_after"] = int(pain_after)
        data["summary"] = summary

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[OK] pain_after güncellendi -> {pain_after}")
    except Exception as e:
        print(f"[ERROR] pain_after yazılamadı: {e}")
# ==============================================================================
# BUTTON
# ==============================================================================
class Button:
    def __init__(self, pos, width, height, text, exercise_name, is_back_button=False):
        self.pos = pos
        self.width = width
        self.height = height
        self.text = text
        self.exercise_name = exercise_name
        self.x, self.y = pos
        self.is_back_button = is_back_button
        self.is_hovered = False

    def draw(self, img):
        bg_color = (RENK_KIRMIZI_GERI if self.is_back_button else RENK_HOVER_BG) if self.is_hovered else RENK_NORMAL_BG
        if self.exercise_name == "CHATBOT_AC":
            bg_color = (50, 150, 50)
        elif self.exercise_name == "PAIN_MINUS":
            bg_color = (120, 80, 80)
        elif self.exercise_name == "PAIN_PLUS":
            bg_color = (80, 120, 80)

        cv2.rectangle(img, self.pos, (self.x + self.width, self.y + self.height), bg_color, -1)
        if self.is_hovered and not self.is_back_button:
            cv2.rectangle(img, self.pos, (self.x + self.width, self.y + self.height), RENK_ACCENT_CYAN, 2)

        img[:] = put_text_tr(img, self.text, (self.x + 10, self.y + 10), 18, RENK_BEYAZ, True)

    def check_hover(self, x, y):
        self.is_hovered = (self.x < x < self.x + self.width and self.y < y < self.y + self.height)

    def check_click(self, x, y):
        return self.x < x < self.x + self.width and self.y < y < self.y + self.height


def create_sidebar_buttons(titles, codes, back_btn=None):
    btns = []
    start_y = 120
    b_height = 45
    b_margin = 10
    for i, (text, code) in enumerate(zip(titles, codes)):
        y_pos = start_y + i * (b_height + b_margin)
        btns.append(Button((20, y_pos), MENU_WIDTH, b_height, text, code))
    if back_btn:
        btns.append(back_btn)
    return btns


# ==============================================================================
# BUTONLAR
# ==============================================================================
back_to_main = Button((20, 650), MENU_WIDTH, 50, "< ANA MENÜ", "MENU_ANA", True)
back_to_omuz = Button((20, 650), MENU_WIDTH, 50, "< OMUZ MENÜSÜ", "MENU_OMUZ", True)

ANA_MENU_BUTTONS = create_sidebar_buttons(
    ["1. Boyun Egzersizleri", "2. Omuz Egzersizleri", "3. Diz Egzersizleri", "4. Kalça Egzersizleri", "5. Bel Egzersizleri"],
    ["MENU_BOYUN", "MENU_OMUZ", "MENU_DIZ", "MENU_KALCA", "MENU_BEL"]
)

chatbot_btn = Button((20, 580), MENU_WIDTH, 50, "💬 AI ASİSTAN", "CHATBOT_AC")
pain_minus_btn = Button((20, 470), 130, 45, "AĞRI -", "PAIN_MINUS")
pain_plus_btn = Button((170, 470), 130, 45, "AĞRI +", "PAIN_PLUS")

ANA_MENU_BUTTONS.append(pain_minus_btn)
ANA_MENU_BUTTONS.append(pain_plus_btn)
ANA_MENU_BUTTONS.append(chatbot_btn)

BOYUN_MENU_BUTTONS = create_sidebar_buttons(
    ["1. Yana Eğilme", "2. Dönme", "3. Öne/Arkaya Eğilme", "4. İzometrik (Öne)", "5. İzometrik (Arkaya)", "6. İzometrik (Yana)", "7. Çember Çizme"],
    ["ROM_LAT", "ROM_ROT", "ROM_FLEKS", "IZO_FLEKS", "IZO_EKST", "IZO_LAT", "ROM_CEMBER"],
    back_to_main
)

OMUZ_MENU_BUTTONS = create_sidebar_buttons(
    ["A. Sopa Egzersizleri", "B. Sallanma (Pendul)", "C. Duvar & Germe"],
    ["MENU_OMUZ_SOPA", "MENU_OMUZ_PEN", "MENU_OMUZ_DUVAR"],
    back_to_main
)

DIZ_MENU_BUTTONS = create_sidebar_buttons(
    ["1. Havlu Ezme (5sn)", "2. Yüzüstü Bükme", "3. Yan Yatarak", "4. Oturarak Uzat", "5. Duvar Squat"],
    ["DIZ_HAVLU_EZME", "DIZ_YUZUSTU_BUKME", "DIZ_YAN_KALDIR", "DIZ_OTUR_UZAT", "DIZ_DUVAR_SQUAT"],
    back_to_main
)

KALCA_MENU_BUTTONS = create_sidebar_buttons(
    ["1. Dizi Göğse Çekme", "2. Düz Bacak Kaldır", "3. Köprü Kurma", "4. Yan Yatarak Açma", "5. Yüzüstü Kaldır", "6. Yan Diz Çekme"],
    ["KALCA_DIZ_CEKME", "KALCA_DUZ_KALDIR", "KALCA_KOPRU", "KALCA_YAN_ACMA", "KALCA_YUZUSTU", "KALCA_YAN_DIZ_CEKME"],
    back_to_main
)

BEL_MENU_BUTTONS = create_sidebar_buttons(
    ["1. Tek Diz Çekme", "2. Çift Diz Çekme", "3. Yarım Mekik", "4. Düz Bacak (SLR)", "5. Köprü (5sn)", "6. Kedi - Deve", "7. Yüzüstü Doğrulma"],
    ["BEL_TEK_DIZ", "BEL_CIFT_DIZ", "BEL_MEKIK", "BEL_SLR", "BEL_KOPRU", "BEL_KEDI_DEVE", "BEL_YUZUSTU"],
    back_to_main
)

OMUZ_SOPA_BUTTONS = create_sidebar_buttons(
    ["1. Yana Açma", "2. Dışa Açma", "3. Öne Açma"],
    ["OMUZ_YANA_ACMA", "OMUZ_DISA_ACMA", "OMUZ_ONE_ACMA"],
    back_to_omuz
)

OMUZ_PEN_BUTTONS = create_sidebar_buttons(
    ["4. Önde Sallama", "5. Yanda Sallama", "6. Çember Çizme"],
    ["OMUZ_PEN_FLEKSIYON", "OMUZ_PEN_ABDUKSIYON", "OMUZ_CEMBER"],
    back_to_omuz
)

OMUZ_DUVAR_BUTTONS = create_sidebar_buttons(
    ["7. Duvara Yana", "8. Duvara Öne", "9. Duvara Geriye", "10. Germe (15sn)"],
    ["OMUZ_DUVAR_YANA", "OMUZ_DUVAR_ONE", "OMUZ_DUVAR_GERIYE", "OMUZ_GERME"],
    back_to_omuz
)


def single_back(target):
    return [Button((20, 650), MENU_WIDTH, 50, "< GERI DON", target, True)]


BOYUN_EX_BTNS = single_back("MENU_BOYUN")
OMUZ_EX_BTNS = single_back("MENU_OMUZ")
DIZ_EX_BTNS = single_back("MENU_DIZ")
KALCA_EX_BTNS = single_back("MENU_KALCA")
BEL_EX_BTNS = single_back("MENU_BEL")

# ==============================================================================
# MOUSE OLAYLARI
# ==============================================================================
def mouse_click_event(event, x, y, flags, param):
    global CURRENT_EXERCISE, PROGRAM_DURUMU, LAST_REP_COUNT, IS_TASK_COMPLETED, IS_SPLIT_MODE
    global SESSION_ERRORS, chatbot_active, CHATBOT_SORU, exercise_recorded, exercise_start_time
    global chat_scroll_offset, chat_max_scroll, chat_ui_info, current_pain_level
    global POST_EXERCISE_PAIN, PENDING_EXERCISE_DATA
    global FRAME_INDEX
    global last_prediction, last_rule_prediction, last_ml_prediction

    if PROGRAM_DURUMU == "ISIM_GIRIS":
        return

    if PROGRAM_DURUMU == "GIRIS_EKRANI":
        if event == cv2.EVENT_LBUTTONDOWN:
            if 440 < x < 840 and 500 < y < 580:
                PROGRAM_DURUMU = "EGZERSIZ_MODU"
                voice_assistant.speak("Egzersiz modu.")
        return

    if chatbot_active and event == cv2.EVENT_LBUTTONDOWN:
        if chat_ui_info:
            up_btn = chat_ui_info.get("up_btn")
            down_btn = chat_ui_info.get("down_btn")

            if up_btn and up_btn[0] <= x <= up_btn[2] and up_btn[1] <= y <= up_btn[3]:
                chat_scroll_offset = min(chat_scroll_offset + 3, chat_max_scroll)
                return

            if down_btn and down_btn[0] <= x <= down_btn[2] and down_btn[1] <= y <= down_btn[3]:
                chat_scroll_offset = max(chat_scroll_offset - 3, 0)
                return

    adjusted_x = x
    if IS_SPLIT_MODE:
        if x < 640:
            return
        adjusted_x = x - 640

    if event == cv2.EVENT_MOUSEMOVE:
        for button in BUTTON_LIST:
            button.check_hover(adjusted_x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        for button in BUTTON_LIST:
            if button.check_click(adjusted_x, y):

                if button.exercise_name == "CHATBOT_AC":
                    chatbot_active = True
                    CHATBOT_SORU = ""
                    voice_assistant.speak("Asistan dinliyor")
                    return

                if button.exercise_name == "PAIN_MINUS":
                    current_pain_level = max(0, current_pain_level - 1)
                    voice_assistant.speak(f"Ağrı seviyesi {current_pain_level}")
                    return

                if button.exercise_name == "PAIN_PLUS":
                    current_pain_level = min(10, current_pain_level + 1)
                    voice_assistant.speak(f"Ağrı seviyesi {current_pain_level}")
                    return

                if button.exercise_name.startswith("MENU_"):
                    CURRENT_EXERCISE = button.exercise_name
                    return

                if "BOYUN" in button.exercise_name or "ROM_" in button.exercise_name or "IZO_" in button.exercise_name:
                    boyun_modulu.reset_boyun_counters()
                elif "OMUZ" in button.exercise_name:
                    omuz_modulu.reset_omuz_counters()
                elif "DIZ" in button.exercise_name:
                    diz_modulu.reset_diz_counters()
                elif "KALCA" in button.exercise_name:
                    kalca_modulu.reset_kalca_counters()
                elif "BEL" in button.exercise_name:
                    bel_modulu.reset_bel_counters()

                CURRENT_EXERCISE = button.exercise_name
                LAST_REP_COUNT = 0
                IS_TASK_COMPLETED = False
                SESSION_ERRORS = []
                exercise_start_time = time.time()
                exercise_recorded = False
                FRAME_INDEX = 0
                last_prediction = None
                last_rule_prediction = None
                last_ml_prediction = None

                # tez log oturumu başlat
                target_angle = get_target_angle_for_exercise(CURRENT_EXERCISE)
                experiment_logger.start_session(
                    patient_name=HASTA_ISMI or "UNKNOWN",
                    exercise_code=CURRENT_EXERCISE,
                    model_name=CURRENT_MODEL_NAME,
                    target_angle=target_angle,
                    target_reps=10
                )

                voice_assistant.speak(f"{button.text} seçildi")
                voice_assistant.speak_instruction(button.exercise_name)
                return


# ==============================================================================
# CAMERA / MEDIAPIPE
# ==============================================================================
POSE_BACKEND_TYPE = "mediapipe"   # "mediapipe", "yolo_pose", "movenet_lightning", "movenet_thunder"

if POSE_BACKEND_TYPE == "mediapipe":
    pose_backend = MediaPipePoseBackend(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
elif POSE_BACKEND_TYPE == "yolo_pose":
    pose_backend = YOLOPoseBackend(model_path="yolov8n-pose.pt")
elif POSE_BACKEND_TYPE == "movenet_lightning":
    pose_backend = MoveNetBackend(variant="lightning")
elif POSE_BACKEND_TYPE == "movenet_thunder":
    pose_backend = MoveNetBackend(variant="thunder")
else:
    raise ValueError(f"Desteklenmeyen backend: {POSE_BACKEND_TYPE}")

CURRENT_MODEL_NAME = pose_backend.model_name

cap_cam = cv2.VideoCapture(0)
cap_cam.set(3, 1280)
cap_cam.set(4, 720)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(WINDOW_NAME, mouse_click_event)

while True:
        
        success_cam, frame_full = cap_cam.read()
        if not success_cam:
            continue
    
        CURRENT_FPS = fps_counter.update()

        frame_full = cv2.flip(frame_full, 1)
        frame_full = cv2.resize(frame_full, (1280, 720))
        key = cv2.waitKey(5) & 0xFF

        # ----------------------------------------------------------------------
        # CHATBOT PANEL
        # ----------------------------------------------------------------------
        if chatbot_active:
            if key == 27:  # ESC
                chatbot_active = False
                if chat_controller and chat_controller.current_session and session_manager is not None:
                    session_manager.save_active_session(chat_controller.current_session)

            elif key == 13:  # Enter
                msg = CHATBOT_SORU.strip().lower()

                if msg == "/plan":
                    if chat_controller:
                        chat_controller.show_saved_plan()
                        chat_scroll_offset = 0
                    CHATBOT_SORU = ""

                elif msg == "/rapor":
                    if chat_controller:
                        chat_controller.show_current_report()
                        chat_scroll_offset = 0
                    CHATBOT_SORU = ""

                elif msg == "/bitir":
                    if chat_controller and chat_controller.current_session and session_manager is not None:
                        final_json = session_manager.finalize_session(chat_controller.current_session)
                        txt_path = kaydet_session_raporu(chat_controller.current_session)
                        pdf_path = export_session_pdf_auto(chat_controller.current_session)

                        extra = ""
                        if last_prediction_text:
                            extra = f"\n\nSon Risk Analizi:\n{last_prediction_text}"

                        chat_controller.messages.append({
                            "role": "assistant",
                            "text": (
                                "Oturum tamamlandı.\n"
                                f"JSON: {final_json}\n"
                                f"TXT rapor: {txt_path}\n"
                                f"PDF rapor: {pdf_path}"
                                f"{extra}"
                            )
                        })

                        chat_controller.current_session = None
                        chat_scroll_offset = 0
                    CHATBOT_SORU = ""

                elif len(CHATBOT_SORU.strip()) > 1 and chat_controller is not None:
                    msg_original = CHATBOT_SORU.strip()
                    lower_msg = msg_original.lower()

                    if "devam edeyim mi" in lower_msg or "devam edebilir miyim" in lower_msg or "risk" in lower_msg:
                        if last_prediction is not None:
                            chat_controller.messages.append({"role": "user", "text": msg_original})
                            chat_controller.messages.append({
                                "role": "assistant",
                                "text": pain_predictor.get_recommendation_text(last_prediction)
                            })
                        else:
                            chat_controller.handle_user_message(msg_original)

                    elif "ağrım kaç" in lower_msg or "agri" in lower_msg:
                        if last_prediction is not None:
                            chat_controller.messages.append({"role": "user", "text": msg_original})
                            chat_controller.messages.append({
                                "role": "assistant",
                                "text": (
                                    f"Son tahmine göre ağrı seviyesi {last_prediction.get('predicted_pain', '?')}/10, "
                                    f"risk seviyesi ise {last_prediction.get('risk_level', '?')}."
                                )
                            })
                        else:
                            chat_controller.handle_user_message(msg_original)
                    else:
                        chat_controller.handle_user_message(msg_original)

                    CHATBOT_SORU = ""
                    chat_scroll_offset = 0

            elif key in (8, 127):  # Backspace
                CHATBOT_SORU = CHATBOT_SORU[:-1]

            elif key == 82:  # Up
                chat_scroll_offset = min(chat_scroll_offset + 2, chat_max_scroll)

            elif key == 84:  # Down
                chat_scroll_offset = max(0, chat_scroll_offset - 2)

            elif 32 <= key <= 126:
                CHATBOT_SORU += chr(key)
            
            if key == ord("q"):
                break

            if chat_controller is not None:
                frame_full, chat_scroll_offset, chat_max_scroll, chat_ui_info = draw_chat_panel(
                    frame=frame_full,
                    messages=chat_controller.messages,
                    input_text=CHATBOT_SORU,
                    scroll_offset=chat_scroll_offset
                )

            cv2.imshow(WINDOW_NAME, frame_full)
            continue

        # ----------------------------------------------------------------------
        # KISAYOLLAR
        # ----------------------------------------------------------------------
        if key == ord("q"):
            break
        elif key == ord("v"):
            voice_assistant.toggle()
        elif key == ord("+") or key == ord("="):
            current_pain_level = min(10, current_pain_level + 1)
        elif key == ord("-"):
            current_pain_level = max(0, current_pain_level - 1)
        
        elif key == ord("x"):
            try:
                pose_backend.close()
            except Exception:
                pass

            if CURRENT_MODEL_NAME == "mediapipe":
                pose_backend = YOLOPoseBackend(model_path="yolov8n-pose.pt")
            elif CURRENT_MODEL_NAME == "yolo_pose":
                pose_backend = MoveNetBackend(variant="lightning")
            else:
                pose_backend = MediaPipePoseBackend(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )

            CURRENT_MODEL_NAME = pose_backend.model_name
            print(f"[INFO] Aktif pose backend: {CURRENT_MODEL_NAME}")
            voice_assistant.speak(f"Model değiştirildi. {CURRENT_MODEL_NAME}")

        # ----------------------------------------------------------------------
        # İSİM GİRİŞ
        # ----------------------------------------------------------------------
        if PROGRAM_DURUMU == "ISIM_GIRIS":
            overlay = frame_full.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 720), (20, 20, 20), -1)
            frame_full = cv2.addWeighted(overlay, 0.9, frame_full, 0.1, 0)

            frame_full = put_text_tr(frame_full, "HASTA KAYIT SISTEMI", (0, 290), 32, RENK_BEYAZ, True, centered=True)
            frame_full = put_text_tr(frame_full, "Adiniz Soyadiniz:", (0, 350), 24, (200, 200, 200), centered=True)
            frame_full = put_text_tr(frame_full, HASTA_ISMI + "|", (0, 400), 40, RENK_ACCENT_CYAN, True, centered=True)

            if key != 255:
                if key == 13 and len(HASTA_ISMI) > 2:
                    PROGRAM_DURUMU = "GIRIS_EKRANI"

                    if session_manager is not None:
                        chat_controller = ChatCoordinator(
                            chatbot=chatbot,
                            planner=planner,
                            session_manager=session_manager,
                            patient_name=HASTA_ISMI
                        )

                    voice_assistant.speak(f"Merhaba {HASTA_ISMI}")


                elif key == 8:
                    HASTA_ISMI = HASTA_ISMI[:-1]
                elif key >= 32:
                    HASTA_ISMI += chr(key).upper()

            cv2.imshow(WINDOW_NAME, frame_full)
            continue

        # ----------------------------------------------------------------------
        # GİRİŞ EKRANI
        # ----------------------------------------------------------------------
        if PROGRAM_DURUMU == "GIRIS_EKRANI":
            overlay = frame_full.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 720), (20, 20, 20), -1)
            frame_full = cv2.addWeighted(overlay, 0.8, frame_full, 0.2, 0)

            frame_full = put_text_tr(frame_full, "FIZYO ASISTAN", (0, 200), 60, RENK_ACCENT_CYAN, True, centered=True)
            frame_full = put_text_tr(frame_full, f"MERHABA, {HASTA_ISMI}", (0, 300), 30, RENK_BEYAZ, True, centered=True)
            frame_full = put_text_tr(frame_full, f"Başlangıç Ağrı Seviyesi: {current_pain_level}/10", (0, 355), 24, RENK_BEYAZ, True, centered=True)
            frame_full = put_text_tr(frame_full, "[-] ve [+] ile değiştirebilirsin", (0, 390), 20, (200, 200, 200), False, centered=True)

            cv2.rectangle(frame_full, (440, 500), (840, 580), RENK_ACCENT_CYAN, 2)
            frame_full = put_text_tr(frame_full, "BASLA", (0, 530), 30, RENK_BEYAZ, True, centered=True)

            cv2.imshow(WINDOW_NAME, frame_full)
            continue
        
        # ----------------------------------------------------------------------
        # EGZERSIZ SONRASI AĞRI GİRİŞİ
        # ----------------------------------------------------------------------
        
        if PROGRAM_DURUMU == "AGRI_SONRASI_GIRIS":
            overlay = frame_full.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 720), (20, 20, 20), -1)
            frame_full = cv2.addWeighted(overlay, 0.9, frame_full, 0.1, 0)

            frame_full = put_text_tr(frame_full, "EGZERSIZ SONRASI DEGERLENDIRME", (0, 220), 34, RENK_ACCENT_CYAN, True, centered=True)
            frame_full = put_text_tr(frame_full, "Agri seviyenizi secin", (0, 300), 26, RENK_BEYAZ, True, centered=True)
            frame_full = put_text_tr(frame_full, f"{POST_EXERCISE_PAIN}/10", (0, 370), 64, RENK_ACCENT_CYAN, True, centered=True)
            frame_full = put_text_tr(frame_full, "[-] ve [+] ile degistir, ENTER ile kaydet", (0, 460), 22, (200, 200, 200), False, centered=True)


            if key != 255:
                if key == ord("+") or key == ord("="):
                    POST_EXERCISE_PAIN = min(10, POST_EXERCISE_PAIN + 1)

                elif key == ord("-"):
                    POST_EXERCISE_PAIN = max(0, POST_EXERCISE_PAIN - 1)

                elif key == 13:
                    if PENDING_EXERCISE_DATA is not None:
                        PENDING_EXERCISE_DATA["record"]["data"]["pain_after"] = POST_EXERCISE_PAIN
                        update_experiment_log_pain_after(CURRENT_SESSION_LOG_PATH, POST_EXERCISE_PAIN)

                        experiment_logger.update_last_session_pain_after(HASTA_ISMI, POST_EXERCISE_PAIN)
                        
                        if LAST_PROGRESS_SUMMARY is not None:
                            LAST_PROGRESS_SUMMARY["pain_after"] = POST_EXERCISE_PAIN
                        

                        try:
                            previous_summary = find_previous_session_summary(
                                logs_dir=os.path.join(BASE_DIR, "experiment_logs"),
                                patient_name=HASTA_ISMI,
                                exercise_code=PENDING_EXERCISE_DATA["exercise_code"],
                                current_log_path=CURRENT_SESSION_LOG_PATH
                            )

                            if LAST_PROGRESS_SUMMARY is not None:
                                LAST_PROGRESS_SUMMARY["pain_after"] = POST_EXERCISE_PAIN

                            if previous_summary is not None and LAST_PROGRESS_SUMMARY is not None:
                                LAST_PROGRESS_REPORT = compare_progress_summaries(
                                    previous_summary,
                                    LAST_PROGRESS_SUMMARY
                                )
                                LAST_PATIENT_FEEDBACK = build_patient_feedback(
                                    LAST_PROGRESS_REPORT,
                                    LAST_PROGRESS_SUMMARY
                                )
                                print("PROGRESS REPORT:", LAST_PROGRESS_REPORT)
                                print("PATIENT FEEDBACK:", LAST_PATIENT_FEEDBACK)
                            else:
                                LAST_PATIENT_FEEDBACK = "Bu egzersiz için ilk kayıt oluşturuldu. Sonraki seanslarda gelişiminizi karşılaştırabileceğiz."
                        except Exception as progress_err:
                            print(f"Progress comparison hatası: {progress_err}")
                            LAST_PATIENT_FEEDBACK = ""

                        from core.session_analysis import classify_session

                        session_type = classify_session(POST_EXERCISE_PAIN)
                        print("SESSION TYPE:", session_type)

                        actual = POST_EXERCISE_PAIN

                        rule_pred = None
                        ml_pred = None

                        if last_rule_prediction:
                            rule_pred = last_rule_prediction.get("predicted_pain")

                        if last_ml_prediction:
                            ml_pred = last_ml_prediction.get("predicted_pain")

                        # Hatalar
                        rule_error = abs(actual - rule_pred) if rule_pred is not None else None
                        ml_error = abs(actual - ml_pred) if ml_pred is not None else None

                        print("GERÇEK:", actual)
                        print("RULE:", rule_pred, "HATA:", rule_error)
                        print("ML:", ml_pred, "HATA:", ml_error)

                        exercise_history.append(PENDING_EXERCISE_DATA["record"])
                        save_history(exercise_history)
                        
                        profile = build_patient_profile(exercise_history, HASTA_ISMI)
                        from core.anomaly import detect_anomaly
                        anomaly_msg = detect_anomaly(profile, PENDING_EXERCISE_DATA["prediction"])
                        adaptive_text = generate_adaptive_recommendation(profile, PENDING_EXERCISE_DATA["prediction"])

                        dashboard_path = generate_progress_dashboard(
                            history=exercise_history,
                            patient_name=HASTA_ISMI,
                            output_dir=os.path.join(BASE_DIR, "dashboards")
                        )

                        print("PATIENT PROFILE:", profile)
                        print("ADAPTIVE RECOMMENDATION:", adaptive_text)
                        print("DASHBOARD PATH:", dashboard_path)

                        if analytics is not None:
                            analytics.record_exercise(
                                HASTA_ISMI,
                                PENDING_EXERCISE_DATA["exercise_code"],
                                {
                                    "reps": PENDING_EXERCISE_DATA["completed_reps"],
                                    "duration": PENDING_EXERCISE_DATA["duration_sec"],
                                    "quality": PENDING_EXERCISE_DATA["quality"],
                                    "predicted_pain": PENDING_EXERCISE_DATA["prediction"].get("predicted_pain", 0),
                                    "pain_after": POST_EXERCISE_PAIN,
                                    "risk_level": PENDING_EXERCISE_DATA["prediction"].get("risk_level", ""),
                                }
                            )

                        if chat_controller is not None:
                            chat_controller.add_exercise_result(
                                exercise_code=PENDING_EXERCISE_DATA["exercise_code"],
                                target_reps=PENDING_EXERCISE_DATA["target_reps"],
                                completed_reps=PENDING_EXERCISE_DATA["completed_reps"],
                                duration_sec=PENDING_EXERCISE_DATA["duration_sec"],
                                status="done"
                            )

                            chat_controller.messages.append({
                                "role": "assistant",
                                "text": (
                                    f"Egzersiz sonrası ağrı seviyesi {POST_EXERCISE_PAIN}/10 olarak kaydedildi.\n"
                                    f"{pain_predictor.get_recommendation_text(PENDING_EXERCISE_DATA['prediction'])}\n"
                                    f"Kişisel öneri: {adaptive_text}"
                                )
                            })
                    
                    PENDING_EXERCISE_DATA = None
                    CURRENT_EXERCISE = "MENU_ANA"
                    LAST_REP_COUNT = 0
                    IS_TASK_COMPLETED = False
                    SESSION_ERRORS = []
                    PROGRAM_DURUMU = "SEANS_GERI_BILDIRIMI"

            cv2.imshow(WINDOW_NAME, frame_full)
            continue
        
        if PROGRAM_DURUMU == "SEANS_GERI_BILDIRIMI":
            overlay = frame_full.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 720), (20, 20, 20), -1)
            frame_full = cv2.addWeighted(overlay, 0.9, frame_full, 0.1, 0)

            frame_full = put_text_tr(
                frame_full,
                "SEANS GERI BILDIRIMI",
                (0, 170),
                34,
                RENK_ACCENT_CYAN,
                True,
                centered=True
            )

            if LAST_PATIENT_FEEDBACK:
                frame_full = draw_multiline_text_centered(
                    frame_full,
                    LAST_PATIENT_FEEDBACK,
                    start_y=260,
                    max_width=950,
                    font_size=22,
                    color=RENK_BEYAZ,
                    line_spacing=34
                )
            else:
                frame_full = put_text_tr(
                    frame_full,
                    "Geri bildirim olusturulamadi.",
                    (0, 320),
                    24,
                    RENK_BEYAZ,
                    False,
                    centered=True
                )

            frame_full = put_text_tr(
                frame_full,
                "Devam etmek icin ENTER",
                (0, 620),
                22,
                (200, 200, 200),
                False,
                centered=True
            )

            if key != 255:
                if key == 13:
                    PROGRAM_DURUMU = "EGZERSIZ_MODU"

            cv2.imshow(WINDOW_NAME, frame_full)
            continue
        # ----------------------------------------------------------------------
        # EGZERSİZ MODU
        # ----------------------------------------------------------------------
        IS_SPLIT_MODE = not CURRENT_EXERCISE.startswith("MENU_")
        if IS_SPLIT_MODE:
            frame_process = cv2.resize(frame_full, (640, 720))
        else:
            frame_process = frame_full.copy()

        
        pose_result = pose_backend.process(frame_process)

        feedback_talimat = ""
        feedback_mesaj = ""
        ekstra_bilgi: Dict[str, Any] = {}
        lm = None

        if pose_result["pose_detected"]:
            pose_backend.draw(frame_process, pose_result)
            lm = pose_result["landmarks"]

            try:
                if "OMUZ" in CURRENT_EXERCISE:
                    draw_visual_protractor(frame_process, lm[23], lm[11], lm[13])
                    draw_visual_protractor(frame_process, lm[24], lm[12], lm[14])
                elif "DIZ" in CURRENT_EXERCISE:
                    draw_visual_protractor(frame_process, lm[23], lm[25], lm[27])
                    draw_visual_protractor(frame_process, lm[24], lm[26], lm[28])
                elif "KALCA" in CURRENT_EXERCISE or "BEL" in CURRENT_EXERCISE:
                    draw_visual_protractor(frame_process, lm[11], lm[23], lm[25])
                    draw_visual_protractor(frame_process, lm[12], lm[24], lm[26])
                elif "BOYUN" in CURRENT_EXERCISE or "ROM_" in CURRENT_EXERCISE or "IZO_" in CURRENT_EXERCISE:
                    draw_visual_protractor(frame_process, lm[0], lm[11], lm[12])
            except Exception:
                pass

        # ----------------------------------------------------------------------
        # MENÜ YÖNETİMİ
        # ----------------------------------------------------------------------
        if CURRENT_EXERCISE == "MENU_ANA":
            BUTTON_LIST = ANA_MENU_BUTTONS
            feedback_talimat = "HOSGELDINIZ"
            feedback_mesaj = f"Ağrı Seviyesi: {current_pain_level}/10"

        elif CURRENT_EXERCISE == "MENU_BOYUN":
            BUTTON_LIST = BOYUN_MENU_BUTTONS
            feedback_mesaj = f"Ağrı Seviyesi: {current_pain_level}/10"

        elif CURRENT_EXERCISE == "MENU_OMUZ":
            BUTTON_LIST = OMUZ_MENU_BUTTONS
            feedback_mesaj = f"Ağrı Seviyesi: {current_pain_level}/10"

        elif CURRENT_EXERCISE == "MENU_OMUZ_SOPA":
            BUTTON_LIST = OMUZ_SOPA_BUTTONS
            feedback_mesaj = f"Ağrı Seviyesi: {current_pain_level}/10"

        elif CURRENT_EXERCISE == "MENU_OMUZ_PEN":
            BUTTON_LIST = OMUZ_PEN_BUTTONS
            feedback_mesaj = f"Ağrı Seviyesi: {current_pain_level}/10"

        elif CURRENT_EXERCISE == "MENU_OMUZ_DUVAR":
            BUTTON_LIST = OMUZ_DUVAR_BUTTONS
            feedback_mesaj = f"Ağrı Seviyesi: {current_pain_level}/10"

        elif CURRENT_EXERCISE == "MENU_DIZ":
            BUTTON_LIST = DIZ_MENU_BUTTONS
            feedback_mesaj = f"Ağrı Seviyesi: {current_pain_level}/10"

        elif CURRENT_EXERCISE == "MENU_KALCA":
            BUTTON_LIST = KALCA_MENU_BUTTONS
            feedback_mesaj = f"Ağrı Seviyesi: {current_pain_level}/10"

        elif CURRENT_EXERCISE == "MENU_BEL":
            BUTTON_LIST = BEL_MENU_BUTTONS
            feedback_mesaj = f"Ağrı Seviyesi: {current_pain_level}/10"

        else:
            if "BOYUN" in CURRENT_EXERCISE or "ROM_" in CURRENT_EXERCISE or "IZO_" in CURRENT_EXERCISE:
                BUTTON_LIST = BOYUN_EX_BTNS
            elif "OMUZ" in CURRENT_EXERCISE:
                BUTTON_LIST = OMUZ_EX_BTNS
            elif "DIZ" in CURRENT_EXERCISE:
                BUTTON_LIST = DIZ_EX_BTNS
            elif "KALCA" in CURRENT_EXERCISE:
                BUTTON_LIST = KALCA_EX_BTNS
            elif "BEL" in CURRENT_EXERCISE:
                BUTTON_LIST = BEL_EX_BTNS

            try:
                if lm is not None:
                    if "BOYUN" in CURRENT_EXERCISE or "ROM_" in CURRENT_EXERCISE or "IZO_" in CURRENT_EXERCISE:
                        feedback_talimat, feedback_mesaj, ekstra_bilgi = boyun_modulu.get_exercise_feedback(
                            CURRENT_EXERCISE,
                            lm,
                            model_name=CURRENT_MODEL_NAME
                        )
                    elif "OMUZ" in CURRENT_EXERCISE:
                        feedback_talimat, feedback_mesaj, ekstra_bilgi = omuz_modulu.get_exercise_feedback(CURRENT_EXERCISE, lm)
                    elif "DIZ" in CURRENT_EXERCISE:
                        feedback_talimat, feedback_mesaj, ekstra_bilgi = diz_modulu.get_exercise_feedback(CURRENT_EXERCISE, lm)
                    elif "KALCA" in CURRENT_EXERCISE:
                        feedback_talimat, feedback_mesaj, ekstra_bilgi = kalca_modulu.get_exercise_feedback(CURRENT_EXERCISE, lm)
                    elif "BEL" in CURRENT_EXERCISE:
                        feedback_talimat, feedback_mesaj, ekstra_bilgi = bel_modulu.get_exercise_feedback(CURRENT_EXERCISE, lm)
            

                if "reps" in ekstra_bilgi and not ("BOYUN" in CURRENT_EXERCISE or "ROM_" in CURRENT_EXERCISE):
                    frame_process = draw_progress_bar(frame_process, ekstra_bilgi["reps"], 10, False)
                
                if "timer" in ekstra_bilgi:
                    frame_process = draw_progress_bar(frame_process, ekstra_bilgi["timer"], 10, True)

                current_angle = None
                current_reps = 0
                current_done = False
                current_confidence = pose_result.get("confidence")

                if isinstance(ekstra_bilgi, dict):
                    if "angle" in ekstra_bilgi:
                        try:
                            current_angle = float(ekstra_bilgi.get("angle"))
                        except Exception:
                            current_angle = None
                    elif "movement_value" in ekstra_bilgi:
                        try:
                            current_angle = float(ekstra_bilgi.get("movement_value"))
                        except Exception:
                            current_angle = None

                    if "reps" in ekstra_bilgi:
                        try:
                            current_reps = int(ekstra_bilgi.get("reps", 0))
                        except Exception:
                            current_reps = 0

                    current_done = bool(ekstra_bilgi.get("done", False))

                if exercise_start_time is not None and not CURRENT_EXERCISE.startswith("MENU_"):
                    target_angle = get_target_angle_for_exercise(CURRENT_EXERCISE)
                    is_complete_for_log = current_done

                    if target_angle is not None and current_angle is not None:
                        if current_angle >= target_angle:
                            is_complete_for_log = True

                    experiment_logger.log_frame(
                        frame_index=FRAME_INDEX,
                        timestamp_sec=time.time() - exercise_start_time,
                        fps=CURRENT_FPS,
                        angle=current_angle,
                        reps=current_reps,
                        is_complete=is_complete_for_log,
                        confidence=current_confidence,
                        extra={
                            "feedback_talimat": feedback_talimat,
                            "feedback_mesaj": feedback_mesaj,
                            "target_angle": target_angle
                        }
                    )
                    FRAME_INDEX += 1

                if feedback_mesaj:
                    feedback_lower = feedback_mesaj.lower()
                    guided_warning_spoken = False

                    if "daha yavaş" in feedback_lower or "yavaş" in feedback_lower:
                        voice_assistant.warn("too_fast")
                        guided_warning_spoken = True
                    elif "merkeze dön" in feedback_lower or "ortaya dön" in feedback_lower:
                        voice_assistant.warn("return_center")
                        guided_warning_spoken = True
                    elif "tamamlayın" in feedback_lower or "biraz daha" in feedback_lower:
                        voice_assistant.warn("not_complete")
                        guided_warning_spoken = True
                    elif "dikkat" in feedback_lower or "form" in feedback_lower or "yanlış" in feedback_lower:
                        voice_assistant.warn("form_error")
                        guided_warning_spoken = True

                    now = time.time()
                    short_guidance = feedback_mesaj.strip()

                    if (
                        not guided_warning_spoken
                        and short_guidance
                        and short_guidance != LAST_SPOKEN_GUIDANCE
                        and (now - LAST_SPOKEN_GUIDANCE_TIME) > 2.5
                    ):
                        voice_assistant.speak(short_guidance)
                        LAST_SPOKEN_GUIDANCE = short_guidance
                        LAST_SPOKEN_GUIDANCE_TIME = now

                    target_reps = int(ekstra_bilgi.get("target_reps", 10)) if isinstance(ekstra_bilgi, dict) else 10
                    curr_reps = None
                    if isinstance(ekstra_bilgi, dict) and "reps" in ekstra_bilgi:
                        try:
                            curr_reps = int(ekstra_bilgi["reps"])
                        except Exception:
                            curr_reps = None

                    if curr_reps is None:
                        nums = re.findall(r"\d+", feedback_mesaj)
                        curr_reps = int(nums[0]) if nums else 0

                    if curr_reps > LAST_REP_COUNT:
                        LAST_REP_COUNT = curr_reps
                        voice_assistant.count_rep(curr_reps, target_reps)

                    done_flag = False
                    if isinstance(ekstra_bilgi, dict):
                        done_flag = bool(ekstra_bilgi.get("done", False))

                    if (curr_reps >= target_reps or done_flag) and not IS_TASK_COMPLETED:
                        IS_TASK_COMPLETED = True
                        voice_assistant.speak("Egzersiz bitti", priority=True)
                        voice_assistant.celebrate("exercise_complete")

                        duration_sec = 0.0
                        if exercise_start_time:
                            duration_sec = time.time() - exercise_start_time

                        angle_value = 0.0
                        try:
                            angle_value = float(ekstra_bilgi.get("angle", 0) or 0)
                        except Exception:
                            angle_value = 0.0

                        quality = calculate_quality_score(
                            exercise_name=CURRENT_EXERCISE,
                            ekstra_bilgi=ekstra_bilgi,
                            feedback_mesaj=feedback_mesaj
                        )

                        result = finalize_exercise_and_predict(
                            exercise_name=CURRENT_EXERCISE,
                            reps=LAST_REP_COUNT,
                            duration_sec=duration_sec,
                            quality=quality,
                            current_pain=current_pain_level,
                            angle=angle_value
                        )

                        prediction = result["prediction"]
                        pending_record = result["record"]

                        PENDING_EXERCISE_DATA = {
                            "record": pending_record,
                            "exercise_code": CURRENT_EXERCISE,
                            "target_reps": target_reps,
                            "completed_reps": LAST_REP_COUNT,
                            "duration_sec": int(duration_sec),
                            "quality": quality,
                            "prediction": prediction
                        }
                        extra_summary = {
                            "exercise_code": str(ekstra_bilgi.get("exercise_code", CURRENT_EXERCISE)),
                            "movement_name": str(ekstra_bilgi.get("movement_name", "movement")),
                            "movement_value": float(ekstra_bilgi.get("movement_value", angle_value) or 0.0),
                            "movement_target": float(ekstra_bilgi.get("movement_target", 0.0) or 0.0),
                            "movement_unit": str(ekstra_bilgi.get("movement_unit", "deg")),
                            "quality_score": float(ekstra_bilgi.get("quality_score", quality) or 0.0),
                            "max_movement_value": float(
                                ekstra_bilgi.get(
                                    "max_movement_value",
                                    ekstra_bilgi.get("movement_value", angle_value)
                                ) or 0.0
                            ),
                            "avg_movement_value": float(ekstra_bilgi.get("movement_value", angle_value) or 0.0),
                            "completed_reps_total": int(
                                ekstra_bilgi.get(
                                    "completed_reps_total",
                                    ekstra_bilgi.get("reps", LAST_REP_COUNT)
                                ) or 0
                            ),
                        }
                        
                        progress_text = (
                            f"Hareket: {extra_summary.get('movement_name', '-')}, "
                            f"Deger: {extra_summary.get('movement_value', 0):.2f} {extra_summary.get('movement_unit', '')}, "
                            f"Hedef: {extra_summary.get('movement_target', 0):.2f}"
                        )

                        if "max_right_value" in extra_summary and "max_left_value" in extra_summary:
                            progress_text += (
                                f" | Sag: {extra_summary.get('max_right_value', 0):.2f}"
                                f" | Sol: {extra_summary.get('max_left_value', 0):.2f}"
                            )

                        print("PROGRESS TEXT:", progress_text)

                        if "max_right_value" in ekstra_bilgi:
                            extra_summary["max_right_value"] = float(ekstra_bilgi.get("max_right_value", 0.0) or 0.0)

                        if "max_left_value" in ekstra_bilgi:
                            extra_summary["max_left_value"] = float(ekstra_bilgi.get("max_left_value", 0.0) or 0.0)

                        if "right_reps" in ekstra_bilgi:
                            extra_summary["right_reps"] = int(ekstra_bilgi.get("right_reps", 0) or 0)

                        if "left_reps" in ekstra_bilgi:
                            extra_summary["left_reps"] = int(ekstra_bilgi.get("left_reps", 0) or 0)

                        if "symmetry_score" in ekstra_bilgi:
                            extra_summary["symmetry_score"] = float(ekstra_bilgi.get("symmetry_score", 0.0) or 0.0)

                        if "hold_time" in ekstra_bilgi:
                            extra_summary["hold_time"] = float(ekstra_bilgi.get("hold_time", 0.0) or 0.0)

                        CURRENT_SESSION_LOG_PATH = experiment_logger.finish_session(
                            completed_reps=LAST_REP_COUNT,
                            duration_sec=duration_sec,
                            pain_before=current_pain_level,
                            pain_after=None,
                            extra_summary=extra_summary
                        )
                        
                        LAST_PROGRESS_REPORT = None
                        LAST_PATIENT_FEEDBACK = ""
                        LAST_PROGRESS_SUMMARY = extra_summary.copy()
                        

                        print("EXPERIMENT LOG PATH:", CURRENT_SESSION_LOG_PATH)
                        print("EXTRA SUMMARY:", extra_summary)

                        POST_EXERCISE_PAIN = current_pain_level
                        PROGRAM_DURUMU = "AGRI_SONRASI_GIRIS"

                        should_continue = pain_predictor.should_continue(prediction)
                        if should_continue:
                            voice_assistant.speak("Risk uygun. Kontrollü devam edebilirsiniz.", priority=True)
                        else:
                            voice_assistant.speak("Risk yüksek. Egzersizi azaltmanız önerilir.", priority=True)

                        if not exercise_recorded and analytics is not None:
                            analytics.record_exercise(
                                HASTA_ISMI,
                                CURRENT_EXERCISE,
                                {
                                    "reps": LAST_REP_COUNT,
                                    "duration": duration_sec,
                                    "quality": quality,
                                    "predicted_pain": prediction.get("predicted_pain", 0),
                                    "risk_level": prediction.get("risk_level", ""),
                                }
                            )

                            if chat_controller is not None:
                                chat_controller.add_exercise_result(
                                    exercise_code=CURRENT_EXERCISE,
                                    target_reps=target_reps,
                                    completed_reps=LAST_REP_COUNT,
                                    duration_sec=int(duration_sec),
                                    status="done"
                                )

                                combined_feedback = pain_predictor.get_recommendation_text(prediction)

                                if LAST_PATIENT_FEEDBACK:
                                    combined_feedback += f"\n\nSeans karşılaştırması:\n{LAST_PATIENT_FEEDBACK}"

                                chat_controller.messages.append({
                                    "role": "assistant",
                                    "text": combined_feedback
                                })

                            exercise_recorded = True

                        CURRENT_EXERCISE = "MENU_ANA"
                        IS_TASK_COMPLETED = False
                        LAST_REP_COUNT = 0
                        SESSION_ERRORS = []

            except Exception as e:
                # Sessiz geçmek yerine log basalım
                print(f"Egzersiz işleme hatası: {e}")

        # ----------------------------------------------------------------------
        # MENÜ OVERLAY
        # ----------------------------------------------------------------------
        if not IS_SPLIT_MODE:
            overlay = frame_process.copy()
            cv2.rectangle(overlay, (0, 0), (MENU_WIDTH + 20, 720), (30, 30, 30), -1)
            frame_process = cv2.addWeighted(overlay, 0.6, frame_process, 0.4, 0)

        # ----------------------------------------------------------------------
        # ÜST BİLGİLER
        # ----------------------------------------------------------------------
        if IS_SPLIT_MODE:
            frame_process = put_text_tr(
                frame_process,
                feedback_talimat,
                (0, 20),
                22,
                RENK_ACCENT_CYAN,
                True,
                centered=True
            )

            frame_process = draw_multiline_text_centered(
                frame_process,
                feedback_mesaj,
                start_y=55,
                max_width=560,
                font_size=18,
                color=RENK_BEYAZ,
                line_spacing=24
            )
        else:
            frame_process = put_text_tr(frame_process, f"{HASTA_ISMI}", (0, 20), 18, RENK_ACCENT_CYAN, True, centered=True)
            frame_process = put_text_tr(frame_process, feedback_talimat, (0, 50), 24, RENK_ACCENT_CYAN, True, centered=True)
            frame_process = put_text_tr(frame_process, feedback_mesaj, (0, 90), 20, RENK_BEYAZ, False, centered=True)
        

        for button in BUTTON_LIST:
            button.draw(frame_process)

        # Son prediction paneli
        if CURRENT_EXERCISE.startswith("MENU_"):
            frame_process = draw_pain_prediction_panel(frame_process, last_prediction)

        # ----------------------------------------------------------------------
        # SPLIT VIDEO
        # ----------------------------------------------------------------------
        if IS_SPLIT_MODE:
            if CURRENT_EXERCISE != PREVIOUS_EXERCISE:
                path = get_exercise_video_path(CURRENT_EXERCISE)
                if cap_video:
                    cap_video.release()
                cap_video = cv2.VideoCapture(path) if path else None
                PREVIOUS_EXERCISE = CURRENT_EXERCISE
                exercise_start_time = time.time()
                exercise_recorded = False
                FRAME_INDEX = 0

            frame_video = np.zeros((720, 640, 3), dtype=np.uint8)
            if cap_video and cap_video.isOpened():
                ret_vid, v_img = cap_video.read()
                if not ret_vid:
                    cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_vid, v_img = cap_video.read()
                if ret_vid:
                    frame_video = cv2.resize(v_img, (640, 720))
            
                angle_text = None
                reps_text = None

                if isinstance(ekstra_bilgi, dict):
                    if "angle" in ekstra_bilgi:
                        try:
                            angle_text = f"Açı: {float(ekstra_bilgi['angle']):.1f}°"
                        except Exception:
                            pass
                    elif "movement_value" in ekstra_bilgi:
                        try:
                            angle_text = f"Açı: {float(ekstra_bilgi['movement_value']):.1f}°"
                        except Exception:
                            pass

                    if isinstance(ekstra_bilgi, dict):
                        if "angle" in ekstra_bilgi:
                            try:
                                angle_text = f"Açı: {float(ekstra_bilgi['angle']):.1f}°"
                            except Exception:
                                pass
                        elif "movement_value" in ekstra_bilgi:
                            try:
                                angle_text = f"Açı: {float(ekstra_bilgi['movement_value']):.1f}°"
                            except Exception:
                                pass

                        if "right_reps" in ekstra_bilgi and "left_reps" in ekstra_bilgi:
                            try:
                                reps_text = f"Sağ: {int(ekstra_bilgi['right_reps'])} | Sol: {int(ekstra_bilgi['left_reps'])}"
                            except Exception:
                                pass
                        elif "reps" in ekstra_bilgi:
                            try:
                                reps_text = f"Tekrar: {int(ekstra_bilgi['reps'])}/10"
                            except Exception:
                                pass
                        elif "timer" in ekstra_bilgi:
                            try:
                                reps_text = f"Süre: {int(ekstra_bilgi['timer'])} sn"
                            except Exception:
                                pass

                frame_video = draw_left_video_overlay(
                    frame_video,
                    patient_name=HASTA_ISMI,
                    pain_level=current_pain_level,
                    fps_value=CURRENT_FPS,
                    model_name=CURRENT_MODEL_NAME,
                    angle_text=angle_text,
                    reps_text=reps_text
                )

            final_view = cv2.hconcat([frame_video, frame_process])
        else:
            final_view = frame_process

        cv2.imshow(WINDOW_NAME, final_view)

# ==============================================================================
# ÇIKIŞ
# ==============================================================================
if chat_controller and chat_controller.current_session and session_manager is not None:
    session_manager.save_active_session(chat_controller.current_session)

cap_cam.release()
if cap_video:
    cap_video.release()

try:
    pose_backend.close()
except Exception:
    pass

cv2.destroyAllWindows()