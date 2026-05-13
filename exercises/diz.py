import numpy as np
import mediapipe as mp
import time

from utils.counter import RepCounter
from utils.timer import DurationTimer
from utils.angles import calculate_angle_3d
from core.progress_metrics import build_progress_payload

mp_pose = mp.solutions.pose.PoseLandmark

# ==================== AYARLAR ====================
HAVLU_EZME_ANGLE = 150
YUZUSTU_FLEX_THRESH = 110
YAN_KALDIR_LIMIT = 150
OTUR_UZAT_THRESH = 150
SQUAT_MIN = 70
SQUAT_MAX = 140

MAX_REPS = 10
HAVLU_HOLD_TIME = 5
YAN_HOLD_TIME = 5
SQUAT_HOLD_TIME = 10

# ==================== SAYAÇLAR ====================
counter_yuzustu_sol = RepCounter("DIZ_YUZUSTU_BUKME", "Sol", threshold_angle=YUZUSTU_FLEX_THRESH, target_reps=MAX_REPS, neutral_threshold=150)
counter_yuzustu_sag = RepCounter("DIZ_YUZUSTU_BUKME", "Sag", threshold_angle=YUZUSTU_FLEX_THRESH, target_reps=MAX_REPS, neutral_threshold=150)

counter_otur_sol = RepCounter("DIZ_OTUR_UZAT", "Sol", threshold_angle=OTUR_UZAT_THRESH, target_reps=MAX_REPS, neutral_threshold=110)
counter_otur_sag = RepCounter("DIZ_OTUR_UZAT", "Sag", threshold_angle=OTUR_UZAT_THRESH, target_reps=MAX_REPS, neutral_threshold=110)

# ==================== GLOBAL STATE ====================
last_hip_y = None

havlu_start_time_sol = None
havlu_start_time_sag = None
havlu_completed_sol = False
havlu_completed_sag = False

yan_start_time_sol = None
yan_start_time_sag = None
yan_completed_sol = False
yan_completed_sag = False

squat_start_time = None
squat_completed = False

# Progress maxima
MAX_YUZUSTU_SOL = 0.0
MAX_YUZUSTU_SAG = 0.0

MAX_OTUR_SOL = 0.0
MAX_OTUR_SAG = 0.0

MAX_HAVLU_SOL = 0.0
MAX_HAVLU_SAG = 0.0

MAX_YAN_KALDIR_SOL = 0.0
MAX_YAN_KALDIR_SAG = 0.0

MAX_SQUAT_VALUE = 0.0

# ==================== RESET ====================
def reset_diz_counters():
    global last_hip_y
    global havlu_start_time_sol, havlu_start_time_sag, havlu_completed_sol, havlu_completed_sag
    global yan_start_time_sol, yan_start_time_sag, yan_completed_sol, yan_completed_sag
    global squat_start_time, squat_completed

    global MAX_YUZUSTU_SOL, MAX_YUZUSTU_SAG
    global MAX_OTUR_SOL, MAX_OTUR_SAG
    global MAX_HAVLU_SOL, MAX_HAVLU_SAG
    global MAX_YAN_KALDIR_SOL, MAX_YAN_KALDIR_SAG
    global MAX_SQUAT_VALUE

    counter_yuzustu_sol.reset()
    counter_yuzustu_sag.reset()
    counter_otur_sol.reset()
    counter_otur_sag.reset()

    last_hip_y = None

    havlu_start_time_sol = None
    havlu_start_time_sag = None
    havlu_completed_sol = False
    havlu_completed_sag = False

    yan_start_time_sol = None
    yan_start_time_sag = None
    yan_completed_sol = False
    yan_completed_sag = False

    squat_start_time = None
    squat_completed = False

    MAX_YUZUSTU_SOL = 0.0
    MAX_YUZUSTU_SAG = 0.0

    MAX_OTUR_SOL = 0.0
    MAX_OTUR_SAG = 0.0

    MAX_HAVLU_SOL = 0.0
    MAX_HAVLU_SAG = 0.0

    MAX_YAN_KALDIR_SOL = 0.0
    MAX_YAN_KALDIR_SAG = 0.0

    MAX_SQUAT_VALUE = 0.0

    print("✅ Diz modülü sıfırlandı.")

# ==================== YARDIMCI FONKSİYONLAR ====================
def get_lm(landmarks, lm_name):
    lm = landmarks[lm_name.value]
    if lm.visibility < 0.25:
        return None
    z_val = getattr(lm, "z", 0.0)
    return [lm.x, lm.y, z_val]


def check_side_lying(l_sh, r_sh):
    y_diff = abs(l_sh[1] - r_sh[1])
    return y_diff > 0.12


def check_prone(l_sh, l_hip):
    y_diff = abs(l_sh[1] - l_hip[1])
    return y_diff < 0.15


def min_pair_reps(sol, sag):
    return int(min(int(sol), int(sag)))


def make_progress_payload(
    *,
    exercise_code,
    reps,
    done,
    movement_name,
    movement_value=0.0,
    movement_target=0.0,
    movement_unit="deg",
    right_value=None,
    left_value=None,
    right_reps=None,
    left_reps=None,
    quality_score=0.75,
    timer=None,
    extra=None
):
    max_val = 0.0
    if right_value is not None or left_value is not None:
        max_val = max(float(right_value or 0.0), float(left_value or 0.0))
    else:
        max_val = float(movement_value or 0.0)

    return build_progress_payload(
        exercise_code=exercise_code,
        reps=int(reps),
        target_reps=MAX_REPS,
        done=bool(done),
        movement_name=movement_name,
        movement_value=float(movement_value or 0.0),
        movement_target=float(movement_target or 0.0),
        movement_unit=movement_unit,
        quality_score=float(quality_score),
        max_movement_value=max_val,
        right_value=right_value,
        left_value=left_value,
        right_reps=right_reps,
        left_reps=left_reps,
        timer=timer,
        extra=extra,
    )

# ==================== ANA FONKSİYON ====================
def get_exercise_feedback(current_exercise, landmarks):
    global last_hip_y
    global havlu_start_time_sol, havlu_start_time_sag, havlu_completed_sol, havlu_completed_sag
    global yan_start_time_sol, yan_start_time_sag, yan_completed_sol, yan_completed_sag
    global squat_start_time, squat_completed

    global MAX_YUZUSTU_SOL, MAX_YUZUSTU_SAG
    global MAX_OTUR_SOL, MAX_OTUR_SAG
    global MAX_HAVLU_SOL, MAX_HAVLU_SAG
    global MAX_YAN_KALDIR_SOL, MAX_YAN_KALDIR_SAG
    global MAX_SQUAT_VALUE

    talimat = ""
    mesaj = ""
    ekstra_bilgi = make_progress_payload(
        exercise_code=current_exercise,
        reps=0,
        done=False,
        movement_name="knee_motion",
        movement_value=0.0,
        movement_target=0.0,
        movement_unit="deg",
        quality_score=0.0
    )

    try:
        l_sh = get_lm(landmarks, mp_pose.LEFT_SHOULDER)
        r_sh = get_lm(landmarks, mp_pose.RIGHT_SHOULDER)
        l_hip = get_lm(landmarks, mp_pose.LEFT_HIP)
        r_hip = get_lm(landmarks, mp_pose.RIGHT_HIP)
        l_knee = get_lm(landmarks, mp_pose.LEFT_KNEE)
        r_knee = get_lm(landmarks, mp_pose.RIGHT_KNEE)
        l_ankle = get_lm(landmarks, mp_pose.LEFT_ANKLE)
        r_ankle = get_lm(landmarks, mp_pose.RIGHT_ANKLE)

        if not l_hip or not r_hip or not l_knee or not r_knee:
            return "⚠️ Gorunmuyorsun", "Bacaklarini goster", make_progress_payload(
                exercise_code=current_exercise,
                reps=0,
                done=False,
                movement_name="knee_motion",
                movement_value=0.0,
                movement_target=0.0,
                movement_unit="deg",
                quality_score=0.0
            )

        # ==================== 1. HAVLU EZME ====================
        if current_exercise == "DIZ_HAVLU_EZME":
            talimat = f"Dizin altina havlu koy, ez ve tut ({HAVLU_HOLD_TIME}sn her bacak)"

            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)

            MAX_HAVLU_SOL = max(MAX_HAVLU_SOL, angle_sol)
            MAX_HAVLU_SAG = max(MAX_HAVLU_SAG, angle_sag)

            # SOL
            if angle_sol > HAVLU_EZME_ANGLE and not havlu_completed_sol:
                if havlu_start_time_sol is None:
                    havlu_start_time_sol = time.time()
                elapsed = time.time() - havlu_start_time_sol
                if elapsed >= HAVLU_HOLD_TIME:
                    havlu_completed_sol = True
                    msg_sol = "✅ TAMAM!"
                else:
                    msg_sol = f"Tut! {int(HAVLU_HOLD_TIME - elapsed)}sn"
            elif havlu_completed_sol:
                msg_sol = "✅ TAMAM!"
            else:
                havlu_start_time_sol = None
                msg_sol = f"Ezmelisin ({int(angle_sol)}°)"

            # SAĞ
            if angle_sag > HAVLU_EZME_ANGLE and not havlu_completed_sag:
                if havlu_start_time_sag is None:
                    havlu_start_time_sag = time.time()
                elapsed = time.time() - havlu_start_time_sag
                if elapsed >= HAVLU_HOLD_TIME:
                    havlu_completed_sag = True
                    msg_sag = "✅ TAMAM!"
                else:
                    msg_sag = f"Tut! {int(HAVLU_HOLD_TIME - elapsed)}sn"
            elif havlu_completed_sag:
                msg_sag = "✅ TAMAM!"
            else:
                havlu_start_time_sag = None
                msg_sag = f"Ezmelisin ({int(angle_sag)}°)"

            done = bool(havlu_completed_sol and havlu_completed_sag)
            reps = 1 if done else 0
            mesaj = f"Sol: {msg_sol} | Sağ: {msg_sag}"

            ekstra_bilgi = make_progress_payload(
                exercise_code="DIZ_HAVLU_EZME",
                reps=reps,
                done=done,
                movement_name="knee_extension_hold",
                movement_value=min(angle_sol, angle_sag),
                movement_target=HAVLU_EZME_ANGLE,
                movement_unit="deg",
                right_value=MAX_HAVLU_SAG,
                left_value=MAX_HAVLU_SOL,
                right_reps=1 if havlu_completed_sag else 0,
                left_reps=1 if havlu_completed_sol else 0,
                quality_score=0.76
            )

        # ==================== 2. YÜZÜSTÜ BÜKME ====================
        elif current_exercise == "DIZ_YUZUSTU_BUKME":
            talimat = "Yuzustu yat, topugunu kalcana cek (Her bacak 10'ar)"

            if not check_prone(l_sh, l_hip):
                mesaj = "⚠️ Yüzüstü yatmalısın!"
                return talimat, mesaj, make_progress_payload(
                    exercise_code="DIZ_YUZUSTU_BUKME",
                    reps=0,
                    done=False,
                    movement_name="knee_flexion_prone",
                    movement_value=0.0,
                    movement_target=YUZUSTU_FLEX_THRESH,
                    movement_unit="deg",
                    quality_score=0.2
                )

            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)

            MAX_YUZUSTU_SOL = max(MAX_YUZUSTU_SOL, angle_sol)
            MAX_YUZUSTU_SAG = max(MAX_YUZUSTU_SAG, angle_sag)

            counter_yuzustu_sol.count(angle_sol)
            counter_yuzustu_sag.count(angle_sag)

            sol_reps = int(counter_yuzustu_sol.rep_count)
            sag_reps = int(counter_yuzustu_sag.rep_count)
            reps = min_pair_reps(sol_reps, sag_reps)
            done = (sol_reps >= MAX_REPS and sag_reps >= MAX_REPS)

            if done:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
            else:
                mesaj = f"{int(angle_sol)}°/{int(angle_sag)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"

            ekstra_bilgi = make_progress_payload(
                exercise_code="DIZ_YUZUSTU_BUKME",
                reps=reps,
                done=done,
                movement_name="knee_flexion_prone",
                movement_value=min(angle_sol, angle_sag),
                movement_target=YUZUSTU_FLEX_THRESH,
                movement_unit="deg",
                right_value=MAX_YUZUSTU_SAG,
                left_value=MAX_YUZUSTU_SOL,
                right_reps=sag_reps,
                left_reps=sol_reps,
                quality_score=0.79
            )

        # ==================== 3. YAN YATARAK KALDIRMA ====================
        elif current_exercise == "DIZ_YAN_KALDIR":
            talimat = f"Yan yat, ustteki bacagi kaldir ({YAN_HOLD_TIME}sn her bacak)"

            if not check_side_lying(l_sh, r_sh):
                mesaj = "⚠️ Yan yatmalısın!"
                return talimat, mesaj, make_progress_payload(
                    exercise_code="DIZ_YAN_KALDIR",
                    reps=0,
                    done=False,
                    movement_name="side_lying_leg_raise_hold",
                    movement_value=0.0,
                    movement_target=YAN_HOLD_TIME,
                    movement_unit="sec",
                    quality_score=0.2
                )

            left_is_up = l_hip[1] < r_hip[1]

            if left_is_up:
                leg_spread = abs(l_ankle[1] - r_ankle[1])
                MAX_YAN_KALDIR_SOL = max(MAX_YAN_KALDIR_SOL, leg_spread)
                is_holding = leg_spread > 0.12

                if is_holding and not yan_completed_sol:
                    if yan_start_time_sol is None:
                        yan_start_time_sol = time.time()
                    elapsed = time.time() - yan_start_time_sol
                    if elapsed >= YAN_HOLD_TIME:
                        yan_completed_sol = True
                        mesaj = "✅ SOL TAMAM!"
                    else:
                        mesaj = f"SOL TUT! {int(YAN_HOLD_TIME - elapsed)}sn"
                elif yan_completed_sol:
                    mesaj = "✅ SOL TAMAM!"
                else:
                    yan_start_time_sol = None
                    mesaj = "Sol: Kaldır"

            else:
                leg_spread = abs(r_ankle[1] - l_ankle[1])
                MAX_YAN_KALDIR_SAG = max(MAX_YAN_KALDIR_SAG, leg_spread)
                is_holding = leg_spread > 0.12

                if is_holding and not yan_completed_sag:
                    if yan_start_time_sag is None:
                        yan_start_time_sag = time.time()
                    elapsed = time.time() - yan_start_time_sag
                    if elapsed >= YAN_HOLD_TIME:
                        yan_completed_sag = True
                        mesaj = "✅ SAĞ TAMAM!"
                    else:
                        mesaj = f"SAĞ TUT! {int(YAN_HOLD_TIME - elapsed)}sn"
                elif yan_completed_sag:
                    mesaj = "✅ SAĞ TAMAM!"
                else:
                    yan_start_time_sag = None
                    mesaj = "Sağ: Kaldır"

            done = bool(yan_completed_sol and yan_completed_sag)
            reps = 1 if done else 0

            ekstra_bilgi = make_progress_payload(
                exercise_code="DIZ_YAN_KALDIR",
                reps=reps,
                done=done,
                movement_name="side_lying_leg_raise_hold",
                movement_value=max(MAX_YAN_KALDIR_SOL, MAX_YAN_KALDIR_SAG),
                movement_target=0.12,
                movement_unit="ratio",
                right_value=MAX_YAN_KALDIR_SAG,
                left_value=MAX_YAN_KALDIR_SOL,
                right_reps=1 if yan_completed_sag else 0,
                left_reps=1 if yan_completed_sol else 0,
                quality_score=0.73
            )

        # ==================== 4. OTURARAK UZATMA ====================
        elif current_exercise == "DIZ_OTUR_UZAT":
            talimat = "Oturarak dizini duzelestir (Her bacak 10'ar)"

            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)

            MAX_OTUR_SOL = max(MAX_OTUR_SOL, angle_sol)
            MAX_OTUR_SAG = max(MAX_OTUR_SAG, angle_sag)

            counter_otur_sol.count(angle_sol)
            counter_otur_sag.count(angle_sag)

            sol_reps = int(counter_otur_sol.rep_count)
            sag_reps = int(counter_otur_sag.rep_count)
            reps = min_pair_reps(sol_reps, sag_reps)
            done = (sol_reps >= MAX_REPS and sag_reps >= MAX_REPS)

            if done:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
            else:
                mesaj = f"{int(angle_sol)}°/{int(angle_sag)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"

            ekstra_bilgi = make_progress_payload(
                exercise_code="DIZ_OTUR_UZAT",
                reps=reps,
                done=done,
                movement_name="seated_knee_extension",
                movement_value=min(angle_sol, angle_sag),
                movement_target=OTUR_UZAT_THRESH,
                movement_unit="deg",
                right_value=MAX_OTUR_SAG,
                left_value=MAX_OTUR_SOL,
                right_reps=sag_reps,
                left_reps=sol_reps,
                quality_score=0.8
            )

        # ==================== 5. DUVAR SQUAT ====================
        elif current_exercise == "DIZ_DUVAR_SQUAT":
            talimat = f"Sirini duvara yasla, çömel ve tut ({SQUAT_HOLD_TIME}sn)"

            angle = calculate_angle_3d(l_hip, l_knee, l_ankle)
            MAX_SQUAT_VALUE = max(MAX_SQUAT_VALUE, angle)

            in_position = SQUAT_MIN < angle < SQUAT_MAX

            if in_position and not squat_completed:
                if squat_start_time is None:
                    squat_start_time = time.time()
                elapsed = time.time() - squat_start_time
                if elapsed >= SQUAT_HOLD_TIME:
                    squat_completed = True
                    mesaj = "✅ HARIKA TAMAMLANDI!"
                    ekstra_bilgi = make_progress_payload(
                        exercise_code="DIZ_DUVAR_SQUAT",
                        reps=1,
                        done=True,
                        movement_name="wall_squat_hold",
                        movement_value=angle,
                        movement_target=SQUAT_MIN,
                        movement_unit="deg",
                        quality_score=0.82,
                        timer=0
                    )
                else:
                    remaining = SQUAT_HOLD_TIME - elapsed
                    mesaj = f"💪 TUT! {int(remaining)}sn | Açı: {int(angle)}°"
                    ekstra_bilgi = make_progress_payload(
                        exercise_code="DIZ_DUVAR_SQUAT",
                        reps=0,
                        done=False,
                        movement_name="wall_squat_hold",
                        movement_value=angle,
                        movement_target=SQUAT_MIN,
                        movement_unit="deg",
                        quality_score=0.76,
                        timer=remaining
                    )

            elif squat_completed:
                mesaj = "✅ TAMAMLANDI!"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="DIZ_DUVAR_SQUAT",
                    reps=1,
                    done=True,
                    movement_name="wall_squat_hold",
                    movement_value=MAX_SQUAT_VALUE,
                    movement_target=SQUAT_MIN,
                    movement_unit="deg",
                    quality_score=0.82,
                    timer=0
                )
            else:
                squat_start_time = None
                if angle >= SQUAT_MAX:
                    mesaj = f"Daha fazla çömel ({int(angle)}°)"
                else:
                    mesaj = f"Çok indin! Biraz kalk ({int(angle)}°)"

                ekstra_bilgi = make_progress_payload(
                    exercise_code="DIZ_DUVAR_SQUAT",
                    reps=0,
                    done=False,
                    movement_name="wall_squat_hold",
                    movement_value=angle,
                    movement_target=SQUAT_MIN,
                    movement_unit="deg",
                    quality_score=0.55
                )

        else:
            mesaj = "Bilinmeyen hareket"

    except Exception as e:
        mesaj = f"❌ Hata: {str(e)}"
        print(f"DIZ MODULU HATA: {e}")

    return talimat, mesaj, ekstra_bilgi