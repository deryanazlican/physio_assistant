import time
import math
import numpy as np
import mediapipe as mp
from collections import deque

from utils.counter import RepCounter
from core.progress_metrics import build_progress_payload

mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark

# ==================== SABITLER ====================
MAX_REPS = 10

LATERAL_THRESH = 10
FLEX_THRESH = 7
EXT_THRESH = 5
NEUTRAL_THRESH = 4

ROTATION_HOLD_TIME = 3.0
ROT_TURN_RATIO = 0.10
ROT_CENTER_RATIO = 0.04

COUNTDOWN_TIME = 3
ISOMETRIC_WORK_TIME = 10
ISOMETRIC_REST_TIME = 5
ISOMETRIC_SETS = 3

CIRCLE_X_THRESH = 0.08
CIRCLE_Y_THRESH = 0.08
CIRCLE_RADIUS_THRESH = 0.05
CIRCLE_MIN_SWEEP = 270.0
CIRCLE_DIR_CHANGE_TOL = 25.0
CIRCLE_COOLDOWN_TIME = 5.0

CALIBRATION_FRAMES = 15

# ==================== BASELINE / KALIBRASYON ====================
baseline_flex = None
baseline_lat = None
baseline_rot = None
baseline_ext_hint = None

flex_samples = deque(maxlen=CALIBRATION_FRAMES)
lat_samples = deque(maxlen=CALIBRATION_FRAMES)
rot_samples = deque(maxlen=CALIBRATION_FRAMES)
ext_hint_samples = deque(maxlen=CALIBRATION_FRAMES)

# ==================== GLOBAL STATE ====================
rot_start_time = None
rotation_held_side = None
rotation_must_return_center = False

izo_state = 0
izo_timer_start = None
izo_current_set = 0
izo_completed = False

circle_count = 0
circle_active = False
circle_prev_angle = None
circle_accumulated = 0.0
circle_direction = None
circle_cooldown_until = 0.0

# ROM_LAT state machine
ROM_LAT_STATE = "CENTER"
ROM_LAT_REPS_RIGHT = 0
ROM_LAT_REPS_LEFT = 0
ROM_LAT_SMOOTH_ANGLE = 0.0

# Progress maxima
ROM_ROT_MAX_RIGHT = 0.0
ROM_ROT_MAX_LEFT = 0.0

ROM_LAT_MAX_RIGHT = 0.0
ROM_LAT_MAX_LEFT = 0.0

ROM_FLEKS_MAX_FORWARD = 0.0
ROM_FLEKS_MAX_BACKWARD = 0.0

# ==================== SAYAÇLAR ====================
lateral_counter_sag = RepCounter("Yana Egilme", "Sag", LATERAL_THRESH, MAX_REPS, NEUTRAL_THRESH)
lateral_counter_sol = RepCounter("Yana Egilme", "Sol", LATERAL_THRESH, MAX_REPS, NEUTRAL_THRESH)

rotasyon_counter_sag = RepCounter("Donme", "Sag", 1, MAX_REPS, 0)
rotasyon_counter_sol = RepCounter("Donme", "Sol", 1, MAX_REPS, 0)

fleksiyon_counter = RepCounter("ROM Fleksiyon", "On", FLEX_THRESH, MAX_REPS, NEUTRAL_THRESH)
ekstansiyon_counter = RepCounter("ROM Ekstansiyon", "Arka", EXT_THRESH, MAX_REPS, NEUTRAL_THRESH)

circle_counter = RepCounter("Boyun Cember", "Tam Tur", 1, MAX_REPS, 0)

# ==================== YARDIMCI ====================
def get_lm(landmarks, lm_name, model_name="mediapipe"):
    lm = landmarks[lm_name.value]
    visibility = getattr(lm, "visibility", 1.0)
    z_val = getattr(lm, "z", 0.0)

    if model_name == "mediapipe":
        if visibility < 0.5:
            return None
    else:
        if visibility < 0.15:
            return None

    return np.array([lm.x, lm.y, z_val], dtype=float)


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _min_pair_reps(a, b):
    return int(min(int(a), int(b)))


def get_head_center(nose, l_ear, r_ear):
    ear_mid = _midpoint(l_ear, r_ear)
    if ear_mid is not None:
        return ear_mid
    return nose


def smooth_value(prev_value: float, new_value: float, alpha: float = 0.25) -> float:
    return (1 - alpha) * prev_value + alpha * new_value


def get_shoulder_center(l_sh, r_sh):
    return np.array([
        (l_sh[0] + r_sh[0]) / 2.0,
        (l_sh[1] + r_sh[1]) / 2.0,
        (l_sh[2] + r_sh[2]) / 2.0
    ], dtype=float)


def get_shoulder_width(l_sh, r_sh):
    w = abs(r_sh[0] - l_sh[0])
    return max(w, 1e-6)


def calculate_lateral_angle(head_center, l_sh, r_sh):
    neck_center = np.array([
        (l_sh[0] + r_sh[0]) / 2.0,
        (l_sh[1] + r_sh[1]) / 2.0
    ], dtype=float)

    vec = np.array([
        head_center[0] - neck_center[0],
        head_center[1] - neck_center[1]
    ], dtype=float)

    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return 0.0

    vertical = np.array([0.0, -1.0], dtype=float)
    cosv = np.dot(vec, vertical) / norm
    cosv = np.clip(cosv, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosv)))

    return angle if head_center[0] > neck_center[0] else -angle


def calculate_flexion_components(nose, head_center, l_sh, r_sh):
    """
    Pozitif = öne eğme
    Negatif = arkaya eğme
    """
    sh_center = get_shoulder_center(l_sh, r_sh)
    sh_w = get_shoulder_width(l_sh, r_sh)

    nose_dy = (nose[1] - sh_center[1]) / sh_w
    head_dy = (head_center[1] - sh_center[1]) / sh_w

    value = (nose_dy * 0.75 + head_dy * 0.25) * 70.0
    return float(value)


def calculate_extension_hint(nose, l_ear, r_ear, l_sh, r_sh):
    sh_w = get_shoulder_width(l_sh, r_sh)
    ear_mid = _midpoint(l_ear, r_ear)
    if ear_mid is None:
        return 0.0

    face_dy = (nose[1] - ear_mid[1]) / sh_w
    return float(face_dy * 60.0)


def calculate_rotation_ratio(nose, l_sh, r_sh):
    cx = (l_sh[0] + r_sh[0]) / 2.0
    sh_w = get_shoulder_width(l_sh, r_sh)
    return float((nose[0] - cx) / sh_w)


def calculate_rotation_angle_deg(nose, l_sh, r_sh):
    cx = (l_sh[0] + r_sh[0]) / 2.0
    cy = (l_sh[1] + r_sh[1]) / 2.0

    dx = nose[0] - cx
    dy = cy - nose[1]

    if abs(dy) < 1e-6:
        return 0.0

    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return float(angle)


def update_baseline(sample_deque, value):
    sample_deque.append(value)
    if len(sample_deque) < CALIBRATION_FRAMES:
        return None
    return float(np.mean(sample_deque))


def calculate_circle_motion(nose, l_sh, r_sh):
    cx = (l_sh[0] + r_sh[0]) / 2.0
    cy = (l_sh[1] + r_sh[1]) / 2.0
    sh_w = get_shoulder_width(l_sh, r_sh)

    dx = (nose[0] - cx) / sh_w
    dy = (nose[1] - cy) / sh_w

    radius = float(np.sqrt(dx * dx + dy * dy))
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if angle < 0:
        angle += 360.0

    return angle, radius, dx, dy


def smallest_angle_diff(current_angle, prev_angle):
    diff = current_angle - prev_angle
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff

# ==================== RESET ====================
def reset_boyun_counters():
    global izo_state, izo_timer_start, izo_completed, izo_current_set
    global rot_start_time, rotation_held_side, rotation_must_return_center
    global baseline_flex, baseline_lat, baseline_rot, baseline_ext_hint
    global circle_count, circle_active, circle_prev_angle
    global circle_accumulated, circle_direction, circle_cooldown_until

    global ROM_LAT_STATE, ROM_LAT_REPS_RIGHT, ROM_LAT_REPS_LEFT, ROM_LAT_SMOOTH_ANGLE
    global ROM_ROT_MAX_RIGHT, ROM_ROT_MAX_LEFT
    global ROM_LAT_MAX_RIGHT, ROM_LAT_MAX_LEFT
    global ROM_FLEKS_MAX_FORWARD, ROM_FLEKS_MAX_BACKWARD

    lateral_counter_sag.reset()
    lateral_counter_sol.reset()
    rotasyon_counter_sag.reset()
    rotasyon_counter_sol.reset()
    fleksiyon_counter.reset()
    ekstansiyon_counter.reset()
    circle_counter.reset()

    izo_state = 0
    izo_timer_start = None
    izo_completed = False
    izo_current_set = 0

    rot_start_time = None
    rotation_held_side = None
    rotation_must_return_center = False

    circle_count = 0
    circle_active = False
    circle_prev_angle = None
    circle_accumulated = 0.0
    circle_direction = None
    circle_cooldown_until = 0.0

    baseline_flex = None
    baseline_lat = None
    baseline_rot = None
    baseline_ext_hint = None

    ROM_LAT_STATE = "CENTER"
    ROM_LAT_REPS_RIGHT = 0
    ROM_LAT_REPS_LEFT = 0
    ROM_LAT_SMOOTH_ANGLE = 0.0

    ROM_ROT_MAX_RIGHT = 0.0
    ROM_ROT_MAX_LEFT = 0.0
    ROM_LAT_MAX_RIGHT = 0.0
    ROM_LAT_MAX_LEFT = 0.0
    ROM_FLEKS_MAX_FORWARD = 0.0
    ROM_FLEKS_MAX_BACKWARD = 0.0

    flex_samples.clear()
    lat_samples.clear()
    rot_samples.clear()
    ext_hint_samples.clear()

    print("✅ Boyun modülü sıfırlandı.")

# ==================== İZOMETRİK ====================
def process_isometric_3sets_neutral(current_angle, tolerance=6):
    global izo_state, izo_timer_start, izo_completed, izo_current_set

    if izo_completed:
        return "✅ HARIKA! 3 SET TAMAMLANDI!", {
            "done": True, "timer": 0, "set": 3,
            "reps": MAX_REPS, "target_reps": MAX_REPS
        }

    if abs(current_angle) > tolerance:
        izo_state = 0
        izo_timer_start = None
        return "❌ Basini sabit tut, yana egme!", {
            "done": False, "timer": 0, "set": izo_current_set,
            "reps": 0, "target_reps": MAX_REPS
        }

    if izo_state == 0:
        izo_state = 1
        izo_timer_start = time.time()
        izo_current_set = 1
        return "⏳ Hazirlan (3 sn)...", {
            "done": False, "timer": COUNTDOWN_TIME, "set": 1,
            "reps": 0, "target_reps": MAX_REPS
        }

    if izo_state == 1:
        elapsed = time.time() - izo_timer_start
        remaining = COUNTDOWN_TIME - elapsed
        if elapsed < COUNTDOWN_TIME:
            return f"⏳ Hazirlan... {int(max(0, remaining))}", {
                "done": False, "timer": max(0, remaining), "set": izo_current_set,
                "reps": 0, "target_reps": MAX_REPS
            }
        izo_state = 2
        izo_timer_start = time.time()
        return "🔥 BASLA! SABIT TUT!", {
            "done": False, "timer": ISOMETRIC_WORK_TIME, "set": izo_current_set,
            "reps": 0, "target_reps": MAX_REPS
        }

    if izo_state == 2:
        elapsed = time.time() - izo_timer_start
        remaining = ISOMETRIC_WORK_TIME - elapsed
        if elapsed < ISOMETRIC_WORK_TIME:
            return f"💪 SABIT TUT! {int(max(0, remaining))}sn | SET {izo_current_set}/3", {
                "done": False, "timer": max(0, remaining), "set": izo_current_set,
                "reps": 0, "target_reps": MAX_REPS
            }

        if izo_current_set >= ISOMETRIC_SETS:
            izo_completed = True
            return "✅ HARIKA! 3 SET TAMAMLANDI!", {
                "done": True, "timer": 0, "set": 3,
                "reps": MAX_REPS, "target_reps": MAX_REPS
            }

        izo_state = 3
        izo_timer_start = time.time()
        return f"😌 DINLEN! {ISOMETRIC_REST_TIME}sn", {
            "done": False, "timer": ISOMETRIC_REST_TIME, "set": izo_current_set,
            "reps": 0, "target_reps": MAX_REPS
        }

    if izo_state == 3:
        elapsed = time.time() - izo_timer_start
        remaining = ISOMETRIC_REST_TIME - elapsed
        if elapsed < ISOMETRIC_REST_TIME:
            return f"😌 Dinlen... {int(max(0, remaining))}sn | SET {izo_current_set}/3", {
                "done": False, "timer": max(0, remaining), "set": izo_current_set,
                "reps": 0, "target_reps": MAX_REPS
            }

        izo_current_set += 1
        izo_state = 2
        izo_timer_start = time.time()
        return f"🔥 SET {izo_current_set} BASLA!", {
            "done": False, "timer": ISOMETRIC_WORK_TIME, "set": izo_current_set,
            "reps": 0, "target_reps": MAX_REPS
        }

    return "", {"done": False, "timer": 0, "set": izo_current_set, "reps": 0, "target_reps": MAX_REPS}


def process_isometric_3sets_active(head_angle):
    global izo_state, izo_timer_start, izo_completed, izo_current_set

    if izo_completed:
        return "✅ HARIKA! 3 SET TAMAMLANDI!", {
            "done": True, "timer": 0, "set": 3,
            "reps": MAX_REPS, "target_reps": MAX_REPS
        }

    if abs(head_angle) < 6:
        izo_state = 0
        izo_timer_start = None
        return "❌ Hafif baski uygula ve pozisyonu koru!", {
            "done": False, "timer": 0, "set": izo_current_set,
            "reps": 0, "target_reps": MAX_REPS
        }

    if izo_state == 0:
        izo_state = 1
        izo_timer_start = time.time()
        izo_current_set = 1
        return "⏳ Hazirlan (3 sn)...", {
            "done": False, "timer": COUNTDOWN_TIME, "set": 1,
            "reps": 0, "target_reps": MAX_REPS
        }

    if izo_state == 1:
        elapsed = time.time() - izo_timer_start
        remaining = COUNTDOWN_TIME - elapsed
        if elapsed < COUNTDOWN_TIME:
            return f"⏳ Hazirlan... {int(max(0, remaining))}", {
                "done": False, "timer": max(0, remaining), "set": izo_current_set,
                "reps": 0, "target_reps": MAX_REPS
            }
        izo_state = 2
        izo_timer_start = time.time()
        return "🔥 BASLA! TUT!", {
            "done": False, "timer": ISOMETRIC_WORK_TIME, "set": izo_current_set,
            "reps": 0, "target_reps": MAX_REPS
        }

    if izo_state == 2:
        elapsed = time.time() - izo_timer_start
        remaining = ISOMETRIC_WORK_TIME - elapsed
        if elapsed < ISOMETRIC_WORK_TIME:
            return f"💪 TUT! {int(max(0, remaining))}sn | SET {izo_current_set}/3", {
                "done": False, "timer": max(0, remaining), "set": izo_current_set,
                "reps": 0, "target_reps": MAX_REPS
            }

        if izo_current_set >= ISOMETRIC_SETS:
            izo_completed = True
            return "✅ HARIKA! 3 SET TAMAMLANDI!", {
                "done": True, "timer": 0, "set": 3,
                "reps": MAX_REPS, "target_reps": MAX_REPS
            }

        izo_state = 3
        izo_timer_start = time.time()
        return f"😌 DINLEN! {ISOMETRIC_REST_TIME}sn", {
            "done": False, "timer": ISOMETRIC_REST_TIME, "set": izo_current_set,
            "reps": 0, "target_reps": MAX_REPS
        }

    if izo_state == 3:
        elapsed = time.time() - izo_timer_start
        remaining = ISOMETRIC_REST_TIME - elapsed
        if elapsed < ISOMETRIC_REST_TIME:
            return f"😌 Dinlen... {int(max(0, remaining))}sn | SET {izo_current_set}/3", {
                "done": False, "timer": max(0, remaining), "set": izo_current_set,
                "reps": 0, "target_reps": MAX_REPS
            }

        izo_current_set += 1
        izo_state = 2
        izo_timer_start = time.time()
        return f"🔥 SET {izo_current_set} BASLA!", {
            "done": False, "timer": ISOMETRIC_WORK_TIME, "set": izo_current_set,
            "reps": 0, "target_reps": MAX_REPS
        }

    return "", {"done": False, "timer": 0, "set": izo_current_set, "reps": 0, "target_reps": MAX_REPS}

# ==================== ANA ====================
def get_exercise_feedback(exercise_name, landmarks, model_name="mediapipe"):
    global rot_start_time, rotation_held_side, rotation_must_return_center
    global baseline_flex, baseline_lat, baseline_rot, baseline_ext_hint
    global ROM_LAT_STATE, ROM_LAT_REPS_RIGHT, ROM_LAT_REPS_LEFT, ROM_LAT_SMOOTH_ANGLE
    global ROM_ROT_MAX_RIGHT, ROM_ROT_MAX_LEFT
    global ROM_LAT_MAX_RIGHT, ROM_LAT_MAX_LEFT
    global ROM_FLEKS_MAX_FORWARD, ROM_FLEKS_MAX_BACKWARD
    global circle_count, circle_active, circle_prev_angle
    global circle_accumulated, circle_direction, circle_cooldown_until

    talimat = ""
    mesaj = ""
    ekstra_bilgi = {}

    try:
        nose = get_lm(landmarks, PoseLandmark.NOSE, model_name=model_name)
        l_sh = get_lm(landmarks, PoseLandmark.LEFT_SHOULDER, model_name=model_name)
        r_sh = get_lm(landmarks, PoseLandmark.RIGHT_SHOULDER, model_name=model_name)
        l_ear = get_lm(landmarks, PoseLandmark.LEFT_EAR, model_name=model_name)
        r_ear = get_lm(landmarks, PoseLandmark.RIGHT_EAR, model_name=model_name)

        if nose is None or l_sh is None or r_sh is None:
            return "⚠️ Gorunmuyorsun", "Kameraya tam karsidan gec", {
                "done": False,
                "reps": 0,
                "target_reps": MAX_REPS
            }

        head_center = get_head_center(nose, l_ear, r_ear)

        # ==================== ROM LAT ====================
        if exercise_name == "ROM_LAT":
            talimat = "Basini saga ve sola eg, her iki tarafa 10'ar tekrar yap"

            raw_lat = calculate_lateral_angle(head_center, l_sh, r_sh)
            if baseline_lat is None:
                baseline_lat = update_baseline(lat_samples, raw_lat)
                return talimat, "Duz bak ve 1 sn sabit kal (kalibrasyon)...", {
                    "done": False,
                    "reps": 0,
                    "target_reps": MAX_REPS
                }

            tilt_angle = raw_lat - baseline_lat

            alpha = 0.35 if model_name in ("yolo_pose", "movenet_lightning", "movenet_thunder") else 0.25
            ROM_LAT_SMOOTH_ANGLE = smooth_value(ROM_LAT_SMOOTH_ANGLE, tilt_angle, alpha=alpha)
            used_angle = ROM_LAT_SMOOTH_ANGLE
            abs_angle = abs(used_angle)

            if used_angle > 0:
                ROM_LAT_MAX_RIGHT = max(ROM_LAT_MAX_RIGHT, abs_angle)
            elif used_angle < 0:
                ROM_LAT_MAX_LEFT = max(ROM_LAT_MAX_LEFT, abs_angle)

            if model_name in ("yolo_pose", "movenet_lightning", "movenet_thunder"):
                target_thresh = 7.0
                neutral_thresh = 3.0
            else:
                target_thresh = float(LATERAL_THRESH)
                neutral_thresh = float(NEUTRAL_THRESH)

            current_side = None
            if used_angle > target_thresh:
                current_side = "RIGHT"
            elif used_angle < -target_thresh:
                current_side = "LEFT"

            done = False
            total_reps = _min_pair_reps(ROM_LAT_REPS_RIGHT, ROM_LAT_REPS_LEFT)

            if ROM_LAT_REPS_RIGHT >= MAX_REPS and ROM_LAT_REPS_LEFT >= MAX_REPS:
                done = True
                mesaj = f"✅ TAMAMLANDI! Sag:{ROM_LAT_REPS_RIGHT}/10 Sol:{ROM_LAT_REPS_LEFT}/10"
            else:
                if ROM_LAT_STATE == "CENTER":
                    if current_side == "RIGHT":
                        ROM_LAT_STATE = "RIGHT"
                        mesaj = f"➡️ Saga egildi | Aci:{abs_angle:.1f}° | Sag:{ROM_LAT_REPS_RIGHT}/10 Sol:{ROM_LAT_REPS_LEFT}/10"
                    elif current_side == "LEFT":
                        ROM_LAT_STATE = "LEFT"
                        mesaj = f"⬅️ Sola egildi | Aci:{abs_angle:.1f}° | Sag:{ROM_LAT_REPS_RIGHT}/10 Sol:{ROM_LAT_REPS_LEFT}/10"
                    else:
                        mesaj = f"Biraz daha yana eg | Aci:{abs_angle:.1f}° | Sag:{ROM_LAT_REPS_RIGHT}/10 Sol:{ROM_LAT_REPS_LEFT}/10"

                elif ROM_LAT_STATE == "RIGHT":
                    if abs_angle <= neutral_thresh:
                        ROM_LAT_REPS_RIGHT += 1
                        ROM_LAT_STATE = "CENTER"
                        mesaj = f"✅ Sag tekrar sayildi | Sag:{ROM_LAT_REPS_RIGHT}/10 Sol:{ROM_LAT_REPS_LEFT}/10"
                    else:
                        mesaj = f"Ortaya don | Aci:{abs_angle:.1f}°"

                elif ROM_LAT_STATE == "LEFT":
                    if abs_angle <= neutral_thresh:
                        ROM_LAT_REPS_LEFT += 1
                        ROM_LAT_STATE = "CENTER"
                        mesaj = f"✅ Sol tekrar sayildi | Sag:{ROM_LAT_REPS_RIGHT}/10 Sol:{ROM_LAT_REPS_LEFT}/10"
                    else:
                        mesaj = f"Ortaya don | Aci:{abs_angle:.1f}°"

                total_reps = _min_pair_reps(ROM_LAT_REPS_RIGHT, ROM_LAT_REPS_LEFT)

            ekstra_bilgi = build_progress_payload(
                exercise_code="ROM_LAT",
                reps=total_reps,
                target_reps=MAX_REPS,
                done=done,
                movement_name="cervical_lateral_flexion",
                movement_value=abs_angle,
                movement_target=float(LATERAL_THRESH),
                movement_unit="deg",
                quality_score=0.7,
                max_movement_value=max(ROM_LAT_MAX_RIGHT, ROM_LAT_MAX_LEFT),
                right_value=ROM_LAT_MAX_RIGHT,
                left_value=ROM_LAT_MAX_LEFT,
                right_reps=ROM_LAT_REPS_RIGHT,
                left_reps=ROM_LAT_REPS_LEFT,
            )

        # ==================== ROM ROT ====================
        elif exercise_name == "ROM_ROT":
            talimat = "Basini saga ve sola cevir, her tarafta 3 sn bekle"

            raw_rot = calculate_rotation_ratio(nose, l_sh, r_sh)
            angle_deg = calculate_rotation_angle_deg(nose, l_sh, r_sh)

            if baseline_rot is None:
                baseline_rot = update_baseline(rot_samples, raw_rot)
                return talimat, "Duz bak ve 1 sn sabit kal (kalibrasyon)...", {
                    "done": False,
                    "reps": 0,
                    "target_reps": MAX_REPS
                }

            rot_ratio = raw_rot - baseline_rot
            abs_ratio = abs(rot_ratio)

            if rot_ratio > 0:
                ROM_ROT_MAX_RIGHT = max(ROM_ROT_MAX_RIGHT, angle_deg)
            elif rot_ratio < 0:
                ROM_ROT_MAX_LEFT = max(ROM_ROT_MAX_LEFT, angle_deg)

            sag_reps = int(rotasyon_counter_sag.rep_count)
            sol_reps = int(rotasyon_counter_sol.rep_count)
            total_reps = _min_pair_reps(sag_reps, sol_reps)

            if sag_reps >= MAX_REPS and sol_reps >= MAX_REPS:
                return talimat, f"✅ TAMAMLANDI! (Sag:{sag_reps} Sol:{sol_reps})", build_progress_payload(
                    exercise_code="ROM_ROT",
                    reps=MAX_REPS,
                    target_reps=MAX_REPS,
                    done=True,
                    movement_name="cervical_rotation",
                    movement_value=float(angle_deg),
                    movement_target=30.0,
                    movement_unit="deg",
                    quality_score=0.8,
                    max_movement_value=max(ROM_ROT_MAX_RIGHT, ROM_ROT_MAX_LEFT),
                    right_value=ROM_ROT_MAX_RIGHT,
                    left_value=ROM_ROT_MAX_LEFT,
                    right_reps=sag_reps,
                    left_reps=sol_reps,
                    timer=0,
                )

            if abs_ratio < ROT_CENTER_RATIO:
                rot_start_time = None
                rotation_held_side = None
                rotation_must_return_center = False
                return talimat, f"✓ MERKEZ | Sag:{sag_reps}/10 Sol:{sol_reps}/10", build_progress_payload(
                    exercise_code="ROM_ROT",
                    reps=total_reps,
                    target_reps=MAX_REPS,
                    done=False,
                    movement_name="cervical_rotation",
                    movement_value=float(angle_deg),
                    movement_target=30.0,
                    movement_unit="deg",
                    quality_score=0.7,
                    max_movement_value=max(ROM_ROT_MAX_RIGHT, ROM_ROT_MAX_LEFT),
                    right_value=ROM_ROT_MAX_RIGHT,
                    left_value=ROM_ROT_MAX_LEFT,
                    right_reps=sag_reps,
                    left_reps=sol_reps,
                    timer=0,
                )

            if rotation_must_return_center:
                return talimat, "Merkeze don", build_progress_payload(
                    exercise_code="ROM_ROT",
                    reps=total_reps,
                    target_reps=MAX_REPS,
                    done=False,
                    movement_name="cervical_rotation",
                    movement_value=float(angle_deg),
                    movement_target=30.0,
                    movement_unit="deg",
                    quality_score=0.7,
                    max_movement_value=max(ROM_ROT_MAX_RIGHT, ROM_ROT_MAX_LEFT),
                    right_value=ROM_ROT_MAX_RIGHT,
                    left_value=ROM_ROT_MAX_LEFT,
                    right_reps=sag_reps,
                    left_reps=sol_reps,
                    timer=0,
                )

            current_side = None
            if rot_ratio > ROT_TURN_RATIO:
                current_side = "sag"
            elif rot_ratio < -ROT_TURN_RATIO:
                current_side = "sol"

            if current_side is None:
                return talimat, "Daha fazla cevir", build_progress_payload(
                    exercise_code="ROM_ROT",
                    reps=total_reps,
                    target_reps=MAX_REPS,
                    done=False,
                    movement_name="cervical_rotation",
                    movement_value=float(angle_deg),
                    movement_target=30.0,
                    movement_unit="deg",
                    quality_score=0.7,
                    max_movement_value=max(ROM_ROT_MAX_RIGHT, ROM_ROT_MAX_LEFT),
                    right_value=ROM_ROT_MAX_RIGHT,
                    left_value=ROM_ROT_MAX_LEFT,
                    right_reps=sag_reps,
                    left_reps=sol_reps,
                    timer=0,
                )

            if rotation_held_side != current_side:
                rotation_held_side = current_side
                rot_start_time = time.time()

            elapsed = time.time() - rot_start_time if rot_start_time else 0.0
            remaining = max(0.0, ROTATION_HOLD_TIME - elapsed)

            if elapsed < ROTATION_HOLD_TIME:
                yon = "SAGA" if current_side == "sag" else "SOLA"
                return talimat, f"{yon} DON ve TUT {remaining:.1f} sn", build_progress_payload(
                    exercise_code="ROM_ROT",
                    reps=total_reps,
                    target_reps=MAX_REPS,
                    done=False,
                    movement_name="cervical_rotation",
                    movement_value=float(angle_deg),
                    movement_target=30.0,
                    movement_unit="deg",
                    quality_score=0.72,
                    max_movement_value=max(ROM_ROT_MAX_RIGHT, ROM_ROT_MAX_LEFT),
                    right_value=ROM_ROT_MAX_RIGHT,
                    left_value=ROM_ROT_MAX_LEFT,
                    right_reps=sag_reps,
                    left_reps=sol_reps,
                    timer=remaining,
                )

            if current_side == "sag":
                rotasyon_counter_sag.rep_count += 1
            else:
                rotasyon_counter_sol.rep_count += 1

            rotation_must_return_center = True
            rot_start_time = None
            rotation_held_side = None

            sag_reps = int(rotasyon_counter_sag.rep_count)
            sol_reps = int(rotasyon_counter_sol.rep_count)

            ekstra_bilgi = build_progress_payload(
                exercise_code="ROM_ROT",
                reps=_min_pair_reps(sag_reps, sol_reps),
                target_reps=MAX_REPS,
                done=False,
                movement_name="cervical_rotation",
                movement_value=float(angle_deg),
                movement_target=30.0,
                movement_unit="deg",
                quality_score=0.75,
                max_movement_value=max(ROM_ROT_MAX_RIGHT, ROM_ROT_MAX_LEFT),
                right_value=ROM_ROT_MAX_RIGHT,
                left_value=ROM_ROT_MAX_LEFT,
                right_reps=sag_reps,
                left_reps=sol_reps,
                timer=0,
            )
            mesaj = "✅ Tekrar sayildi, merkeze don"

        # ==================== ROM FLEKS ====================
        elif exercise_name == "ROM_FLEKS":
            talimat = "Ceneyi gogse gotur, sonra tavana bak (Her ikisi 10'ar)"

            raw_flex = calculate_flexion_components(nose, head_center, l_sh, r_sh)
            raw_ext_hint = calculate_extension_hint(nose, l_ear, r_ear, l_sh, r_sh)

            if baseline_flex is None:
                baseline_flex = update_baseline(flex_samples, raw_flex)
                return talimat, "Duz bak ve 1 sn sabit kal (kalibrasyon)...", {
                    "done": False,
                    "reps": 0,
                    "target_reps": MAX_REPS
                }

            if baseline_ext_hint is None:
                baseline_ext_hint = update_baseline(ext_hint_samples, raw_ext_hint)
                return talimat, "Duz bak ve 1 sn sabit kal (kalibrasyon)...", {
                    "done": False,
                    "reps": 0,
                    "target_reps": MAX_REPS
                }

            flex_value = raw_flex - baseline_flex
            ext_hint = raw_ext_hint - baseline_ext_hint

            forward_angle = flex_value
            extension_score = (-flex_value * 0.7) + (-ext_hint * 0.3)
            abs_angle = max(abs(forward_angle), abs(extension_score))

            if forward_angle > 0:
                ROM_FLEKS_MAX_FORWARD = max(ROM_FLEKS_MAX_FORWARD, abs(forward_angle))
            if extension_score > 0:
                ROM_FLEKS_MAX_BACKWARD = max(ROM_FLEKS_MAX_BACKWARD, abs(extension_score))

            flex_reps = int(fleksiyon_counter.rep_count)
            ext_reps = int(ekstansiyon_counter.rep_count)

            if flex_reps >= MAX_REPS and ext_reps >= MAX_REPS:
                return talimat, f"✅ TAMAMLANDI! (One:{flex_reps} Arkaya:{ext_reps})", build_progress_payload(
                    exercise_code="ROM_FLEKS",
                    reps=MAX_REPS,
                    target_reps=MAX_REPS,
                    done=True,
                    movement_name="cervical_flexion_extension",
                    movement_value=abs_angle,
                    movement_target=float(max(FLEX_THRESH, EXT_THRESH)),
                    movement_unit="deg",
                    quality_score=0.8,
                    max_movement_value=max(ROM_FLEKS_MAX_FORWARD, ROM_FLEKS_MAX_BACKWARD),
                    right_value=ROM_FLEKS_MAX_FORWARD,
                    left_value=ROM_FLEKS_MAX_BACKWARD,
                    right_reps=flex_reps,
                    left_reps=ext_reps,
                )

            if abs(forward_angle) < NEUTRAL_THRESH and abs(extension_score) < NEUTRAL_THRESH:
                fleksiyon_counter.count(0)
                ekstansiyon_counter.count(0)
                mesaj = f"✓ MERKEZ | One:{flex_reps}/10 Arkaya:{ext_reps}/10"
            elif forward_angle > FLEX_THRESH:
                fleksiyon_counter.count(abs(forward_angle))
                mesaj = f"⬇️ ONE EG {int(abs(forward_angle))}° | One:{int(fleksiyon_counter.rep_count)}/10"
            elif extension_score > EXT_THRESH:
                ekstansiyon_counter.count(extension_score)
                mesaj = f"⬆️ ARKAYA GIT {int(extension_score)}° | Arkaya:{int(ekstansiyon_counter.rep_count)}/10"
            else:
                mesaj = f"Daha belirgin one/arkaya eg | One:{flex_reps}/10 Arkaya:{ext_reps}/10"

            ekstra_bilgi = build_progress_payload(
                exercise_code="ROM_FLEKS",
                reps=_min_pair_reps(fleksiyon_counter.rep_count, ekstansiyon_counter.rep_count),
                target_reps=MAX_REPS,
                done=False,
                movement_name="cervical_flexion_extension",
                movement_value=abs_angle,
                movement_target=float(max(FLEX_THRESH, EXT_THRESH)),
                movement_unit="deg",
                quality_score=0.75,
                max_movement_value=max(ROM_FLEKS_MAX_FORWARD, ROM_FLEKS_MAX_BACKWARD),
                right_value=ROM_FLEKS_MAX_FORWARD,
                left_value=ROM_FLEKS_MAX_BACKWARD,
                right_reps=int(fleksiyon_counter.rep_count),
                left_reps=int(ekstansiyon_counter.rep_count),
            )

        # ==================== ROM CEMBER ====================
        elif exercise_name == "ROM_CEMBER":
            talimat = "Basini daire cizer gibi yavas ve genis sekilde döndür"

            angle, radius, dx, dy = calculate_circle_motion(nose, l_sh, r_sh)
            now = time.time()

            if circle_count >= MAX_REPS:
                return talimat, f"✅ TAMAMLANDI! ({MAX_REPS} tur)", {
                    "reps": circle_count,
                    "target_reps": MAX_REPS,
                    "done": True,
                    "angle": angle,
                    "radius": radius
                }

            if now < circle_cooldown_until:
                remaining = circle_cooldown_until - now
                circle_active = False
                circle_prev_angle = None
                circle_accumulated = 0.0
                circle_direction = None

                return talimat, f"⏳ {remaining:.1f} sn bekle, sonra tekrar ciz | Tur:{circle_count}/10", {
                    "reps": circle_count,
                    "target_reps": MAX_REPS,
                    "done": False,
                    "angle": angle,
                    "radius": radius,
                    "timer": remaining
                }

            if radius < CIRCLE_RADIUS_THRESH:
                circle_active = False
                circle_prev_angle = None
                circle_accumulated = 0.0
                circle_direction = None

                return talimat, f"Daha genis daire ciz | Tur:{circle_count}/10", {
                    "reps": circle_count,
                    "target_reps": MAX_REPS,
                    "done": False,
                    "angle": angle,
                    "radius": radius
                }

            if not circle_active:
                circle_active = True
                circle_prev_angle = angle
                circle_accumulated = 0.0
                circle_direction = None

                return talimat, f"Basla... daireyi tamamla | Tur:{circle_count}/10", {
                    "reps": circle_count,
                    "target_reps": MAX_REPS,
                    "done": False,
                    "angle": angle,
                    "radius": radius
                }

            diff = smallest_angle_diff(angle, circle_prev_angle)
            circle_prev_angle = angle

            if abs(diff) < 2.0:
                return talimat, f"Devam et | Tur:{circle_count}/10", {
                    "reps": circle_count,
                    "target_reps": MAX_REPS,
                    "done": False,
                    "angle": angle,
                    "radius": radius,
                    "progress_deg": circle_accumulated
                }

            current_dir = "CCW" if diff > 0 else "CW"

            if circle_direction is None:
                circle_direction = current_dir
                circle_accumulated += abs(diff)
            else:
                if current_dir == circle_direction:
                    circle_accumulated += abs(diff)
                else:
                    if abs(diff) > CIRCLE_DIR_CHANGE_TOL:
                        circle_accumulated = 0.0
                        circle_direction = current_dir
                    else:
                        circle_accumulated += abs(diff)

            if circle_accumulated >= CIRCLE_MIN_SWEEP:
                circle_count += 1
                circle_cooldown_until = now + CIRCLE_COOLDOWN_TIME
                circle_active = False
                circle_prev_angle = None
                circle_accumulated = 0.0
                circle_direction = None

                return talimat, f"✅ TUR {circle_count}! {int(CIRCLE_COOLDOWN_TIME)} sn bekle", {
                    "reps": circle_count,
                    "target_reps": MAX_REPS,
                    "done": (circle_count >= MAX_REPS),
                    "angle": angle,
                    "radius": radius,
                    "timer": CIRCLE_COOLDOWN_TIME
                }

            return talimat, f"Devam et | Yon:{circle_direction or '-'} | Tur:{circle_count}/10", {
                "reps": circle_count,
                "target_reps": MAX_REPS,
                "done": False,
                "angle": angle,
                "radius": radius,
                "progress_deg": circle_accumulated
            }

        # ==================== IZO FLEKS ====================
        elif exercise_name == "IZO_FLEKS":
            talimat = "Elleri alnina koy ve one it (3x10sn)"

            raw_flex = calculate_flexion_components(nose, head_center, l_sh, r_sh)
            if baseline_flex is None:
                baseline_flex = update_baseline(flex_samples, raw_flex)
                return talimat, "Duz bak ve 1 sn sabit kal (kalibrasyon)...", {
                    "done": False,
                    "reps": 0,
                    "target_reps": MAX_REPS
                }

            angle = raw_flex - baseline_flex
            mesaj, ekstra_bilgi = process_isometric_3sets_active(angle)

        # ==================== IZO EKST ====================
        elif exercise_name == "IZO_EKST":
            talimat = "Elleri ensene koy ve geriye it (3x10sn)"

            raw_flex = calculate_flexion_components(nose, head_center, l_sh, r_sh)
            if baseline_flex is None:
                baseline_flex = update_baseline(flex_samples, raw_flex)
                return talimat, "Duz bak ve 1 sn sabit kal (kalibrasyon)...", {
                    "done": False,
                    "reps": 0,
                    "target_reps": MAX_REPS
                }

            angle = raw_flex - baseline_flex
            mesaj, ekstra_bilgi = process_isometric_3sets_active(angle)

        # ==================== IZO LAT ====================
        elif exercise_name == "IZO_LAT":
            talimat = "Elini sakagina koy ve yana it, ama basini sabit tut (3x10sn)"

            raw_lat = calculate_lateral_angle(head_center, l_sh, r_sh)
            if baseline_lat is None:
                baseline_lat = update_baseline(lat_samples, raw_lat)
                return talimat, "Duz bak ve basini sabit tut (kalibrasyon)...", {
                    "done": False,
                    "reps": 0,
                    "target_reps": MAX_REPS
                }

            angle = raw_lat - baseline_lat
            mesaj, ekstra_bilgi = process_isometric_3sets_neutral(angle, tolerance=6)

        else:
            talimat = "⚠️ Bilinmeyen Hareket"
            mesaj = f"Gelen: {exercise_name}"
            ekstra_bilgi = {"done": False, "reps": 0, "target_reps": MAX_REPS}

    except Exception as e:
        mesaj = f"❌ Hata: {str(e)}"
        print(f"BOYUN MODULU HATA: {e}")
        ekstra_bilgi = {"done": False, "reps": 0, "target_reps": MAX_REPS}

    if "reps" not in ekstra_bilgi:
        ekstra_bilgi["reps"] = 0
    if "target_reps" not in ekstra_bilgi:
        ekstra_bilgi["target_reps"] = MAX_REPS
    if "done" not in ekstra_bilgi:
        ekstra_bilgi["done"] = False

    return talimat, mesaj, ekstra_bilgi