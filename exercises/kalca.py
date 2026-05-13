import numpy as np
import mediapipe as mp
import time

from utils.counter import RepCounter
from utils.angles import calculate_angle_3d
from core.progress_metrics import build_progress_payload

mp_pose = mp.solutions.pose.PoseLandmark

# ==================== AYARLAR ====================
DIZ_CEKME_THRESH = 100
DUZ_KALDIR_THRESH = 140
KOPRU_THRESH = 150
YAN_HOLD_TIME = 5
YUZUSTU_HOLD_TIME = 5

MAX_REPS = 10

# ==================== SAYAÇLAR ====================
counter_diz_cekme_sol = RepCounter("KALCA_DIZ_CEKME", "Sol", threshold_angle=DIZ_CEKME_THRESH, target_reps=MAX_REPS, neutral_threshold=160)
counter_diz_cekme_sag = RepCounter("KALCA_DIZ_CEKME", "Sag", threshold_angle=DIZ_CEKME_THRESH, target_reps=MAX_REPS, neutral_threshold=160)

counter_duz_kaldir_sol = RepCounter("KALCA_DUZ_KALDIR", "Sol", threshold_angle=DUZ_KALDIR_THRESH, target_reps=MAX_REPS, neutral_threshold=170)
counter_duz_kaldir_sag = RepCounter("KALCA_DUZ_KALDIR", "Sag", threshold_angle=DUZ_KALDIR_THRESH, target_reps=MAX_REPS, neutral_threshold=170)

counter_kopru = RepCounter("KALCA_KOPRU", "Kalca", threshold_angle=KOPRU_THRESH, target_reps=MAX_REPS, neutral_threshold=140)

counter_yan_diz_sol = RepCounter("KALCA_YAN_DIZ_CEKME", "Sol", threshold_angle=DIZ_CEKME_THRESH, target_reps=MAX_REPS, neutral_threshold=160)
counter_yan_diz_sag = RepCounter("KALCA_YAN_DIZ_CEKME", "Sag", threshold_angle=DIZ_CEKME_THRESH, target_reps=MAX_REPS, neutral_threshold=160)

# ==================== GLOBAL STATE ====================
yan_start_time_sol = None
yan_start_time_sag = None
yan_completed_sol = False
yan_completed_sag = False

yuzustu_start_time_sol = None
yuzustu_start_time_sag = None
yuzustu_completed_sol = False
yuzustu_completed_sag = False

# Progress maxima
MAX_DIZ_CEKME_SOL = 0.0
MAX_DIZ_CEKME_SAG = 0.0

MAX_DUZ_KALDIR_SOL = 0.0
MAX_DUZ_KALDIR_SAG = 0.0

MAX_KOPRU = 0.0

MAX_YAN_ACMA_SOL = 0.0
MAX_YAN_ACMA_SAG = 0.0

MAX_YUZUSTU_SOL = 0.0
MAX_YUZUSTU_SAG = 0.0

MAX_YAN_DIZ_CEKME_SOL = 0.0
MAX_YAN_DIZ_CEKME_SAG = 0.0

# ==================== RESET ====================
def reset_kalca_counters():
    global yan_start_time_sol, yan_start_time_sag, yan_completed_sol, yan_completed_sag
    global yuzustu_start_time_sol, yuzustu_start_time_sag, yuzustu_completed_sol, yuzustu_completed_sag

    global MAX_DIZ_CEKME_SOL, MAX_DIZ_CEKME_SAG
    global MAX_DUZ_KALDIR_SOL, MAX_DUZ_KALDIR_SAG
    global MAX_KOPRU
    global MAX_YAN_ACMA_SOL, MAX_YAN_ACMA_SAG
    global MAX_YUZUSTU_SOL, MAX_YUZUSTU_SAG
    global MAX_YAN_DIZ_CEKME_SOL, MAX_YAN_DIZ_CEKME_SAG

    counter_diz_cekme_sol.reset()
    counter_diz_cekme_sag.reset()
    counter_duz_kaldir_sol.reset()
    counter_duz_kaldir_sag.reset()
    counter_kopru.reset()
    counter_yan_diz_sol.reset()
    counter_yan_diz_sag.reset()

    yan_start_time_sol = None
    yan_start_time_sag = None
    yan_completed_sol = False
    yan_completed_sag = False

    yuzustu_start_time_sol = None
    yuzustu_start_time_sag = None
    yuzustu_completed_sol = False
    yuzustu_completed_sag = False

    MAX_DIZ_CEKME_SOL = 0.0
    MAX_DIZ_CEKME_SAG = 0.0

    MAX_DUZ_KALDIR_SOL = 0.0
    MAX_DUZ_KALDIR_SAG = 0.0

    MAX_KOPRU = 0.0

    MAX_YAN_ACMA_SOL = 0.0
    MAX_YAN_ACMA_SAG = 0.0

    MAX_YUZUSTU_SOL = 0.0
    MAX_YUZUSTU_SAG = 0.0

    MAX_YAN_DIZ_CEKME_SOL = 0.0
    MAX_YAN_DIZ_CEKME_SAG = 0.0

    print("✅ Kalça modülü sıfırlandı.")

# ==================== YARDIMCI FONKSİYONLAR ====================
def get_lm(landmarks, lm_name):
    lm = landmarks[lm_name.value]
    if lm.visibility < 0.25:
        return None
    z_val = getattr(lm, "z", 0.0)
    return [lm.x, lm.y, z_val]


def check_side_lying(l_sh, r_sh):
    y_diff = abs(l_sh[1] - r_sh[1])
    return y_diff > 0.10


def get_top_leg(l_hip, r_hip):
    return "SOL" if l_hip[1] < r_hip[1] else "SAG"


def check_prone(l_sh, l_hip):
    return abs(l_sh[1] - l_hip[1]) < 0.15


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
    global yan_start_time_sol, yan_start_time_sag, yan_completed_sol, yan_completed_sag
    global yuzustu_start_time_sol, yuzustu_start_time_sag, yuzustu_completed_sol, yuzustu_completed_sag

    global MAX_DIZ_CEKME_SOL, MAX_DIZ_CEKME_SAG
    global MAX_DUZ_KALDIR_SOL, MAX_DUZ_KALDIR_SAG
    global MAX_KOPRU
    global MAX_YAN_ACMA_SOL, MAX_YAN_ACMA_SAG
    global MAX_YUZUSTU_SOL, MAX_YUZUSTU_SAG
    global MAX_YAN_DIZ_CEKME_SOL, MAX_YAN_DIZ_CEKME_SAG

    talimat = ""
    mesaj = ""
    ekstra_bilgi = make_progress_payload(
        exercise_code=current_exercise,
        reps=0,
        done=False,
        movement_name="hip_motion",
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

        if not l_hip or not r_hip or not l_knee:
            return "⚠️ Gorunmuyorsun", "Kameraya gec", make_progress_payload(
                exercise_code=current_exercise,
                reps=0,
                done=False,
                movement_name="hip_motion",
                movement_value=0.0,
                movement_target=0.0,
                movement_unit="deg",
                quality_score=0.0
            )

        # ==================== 1. DİZİ GÖĞSE ÇEKME ====================
        if current_exercise == "KALCA_DIZ_CEKME":
            talimat = "Sirtustu yat, dizini gogsune cek (Her bacak 10'ar)"

            angle_sol = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_sag = calculate_angle_3d(r_sh, r_hip, r_knee)

            MAX_DIZ_CEKME_SOL = max(MAX_DIZ_CEKME_SOL, angle_sol)
            MAX_DIZ_CEKME_SAG = max(MAX_DIZ_CEKME_SAG, angle_sag)

            counter_diz_cekme_sol.count(angle_sol)
            counter_diz_cekme_sag.count(angle_sag)

            sol_reps = int(counter_diz_cekme_sol.rep_count)
            sag_reps = int(counter_diz_cekme_sag.rep_count)
            reps = min_pair_reps(sol_reps, sag_reps)
            done = (sol_reps >= MAX_REPS and sag_reps >= MAX_REPS)

            if done:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
            else:
                mesaj = f"{int(angle_sol)}°/{int(angle_sag)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"

            ekstra_bilgi = make_progress_payload(
                exercise_code="KALCA_DIZ_CEKME",
                reps=reps,
                done=done,
                movement_name="hip_flexion_knee_to_chest",
                movement_value=min(angle_sol, angle_sag),
                movement_target=DIZ_CEKME_THRESH,
                movement_unit="deg",
                right_value=MAX_DIZ_CEKME_SAG,
                left_value=MAX_DIZ_CEKME_SOL,
                right_reps=sag_reps,
                left_reps=sol_reps,
                quality_score=0.8
            )

        # ==================== 2. DÜZ BACAK KALDIRMA ====================
        elif current_exercise == "KALCA_DUZ_KALDIR":
            talimat = "Dizini BUKMEDEN bacagini kaldir (Her bacak 10'ar)"

            knee_angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            knee_angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)

            hip_angle_sol = calculate_angle_3d(l_sh, l_hip, l_knee)
            hip_angle_sag = calculate_angle_3d(r_sh, r_hip, r_knee)

            MAX_DUZ_KALDIR_SOL = max(MAX_DUZ_KALDIR_SOL, hip_angle_sol)
            MAX_DUZ_KALDIR_SAG = max(MAX_DUZ_KALDIR_SAG, hip_angle_sag)

            sol_reps = int(counter_duz_kaldir_sol.rep_count)
            sag_reps = int(counter_duz_kaldir_sag.rep_count)

            msg_sol = ""
            msg_sag = ""

            if knee_angle_sol < 140:
                msg_sol = "Dizi duzelt!"
            else:
                msg_sol = counter_duz_kaldir_sol.count(hip_angle_sol)

            if knee_angle_sag < 140:
                msg_sag = "Dizi duzelt!"
            else:
                msg_sag = counter_duz_kaldir_sag.count(hip_angle_sag)

            sol_reps = int(counter_duz_kaldir_sol.rep_count)
            sag_reps = int(counter_duz_kaldir_sag.rep_count)

            reps = min_pair_reps(sol_reps, sag_reps)
            done = (sol_reps >= MAX_REPS and sag_reps >= MAX_REPS)

            if done:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
            else:
                mesaj = f"Sol:{sol_reps}/10 ({msg_sol}) | Sağ:{sag_reps}/10 ({msg_sag})"

            ekstra_bilgi = make_progress_payload(
                exercise_code="KALCA_DUZ_KALDIR",
                reps=reps,
                done=done,
                movement_name="straight_leg_raise",
                movement_value=min(hip_angle_sol, hip_angle_sag),
                movement_target=DUZ_KALDIR_THRESH,
                movement_unit="deg",
                right_value=MAX_DUZ_KALDIR_SAG,
                left_value=MAX_DUZ_KALDIR_SOL,
                right_reps=sag_reps,
                left_reps=sol_reps,
                quality_score=0.79,
                extra={
                    "left_knee_extension_angle": float(knee_angle_sol),
                    "right_knee_extension_angle": float(knee_angle_sag),
                }
            )

        # ==================== 3. KÖPRÜ KURMA ====================
        elif current_exercise == "KALCA_KOPRU":
            talimat = "Kalçani havaya kaldir ve sik (10 tekrar)"

            angle_sol = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_sag = calculate_angle_3d(r_sh, r_hip, r_knee)
            avg_angle = (angle_sol + angle_sag) / 2.0

            MAX_KOPRU = max(MAX_KOPRU, avg_angle)

            counter_kopru.count(avg_angle)
            reps = int(counter_kopru.rep_count)
            done = reps >= MAX_REPS

            if done:
                mesaj = f"✅ TAMAMLANDI! ({MAX_REPS} tekrar)"
            else:
                mesaj = f"{int(avg_angle)}° | {reps}/10"

            ekstra_bilgi = make_progress_payload(
                exercise_code="KALCA_KOPRU",
                reps=reps,
                done=done,
                movement_name="bridge_hip_extension",
                movement_value=avg_angle,
                movement_target=KOPRU_THRESH,
                movement_unit="deg",
                right_value=angle_sag,
                left_value=angle_sol,
                right_reps=reps,
                left_reps=reps,
                quality_score=0.81
            )

        # ==================== 4. YAN YATARAK AÇMA ====================
        elif current_exercise == "KALCA_YAN_ACMA":
            talimat = f"Yan yat, ustteki bacagi kaldir ({YAN_HOLD_TIME}sn her bacak)"

            if not check_side_lying(l_sh, r_sh):
                mesaj = "⚠️ Yan yatmalısın!"
                return talimat, mesaj, make_progress_payload(
                    exercise_code="KALCA_YAN_ACMA",
                    reps=0,
                    done=False,
                    movement_name="side_lying_hip_abduction_hold",
                    movement_value=0.0,
                    movement_target=YAN_HOLD_TIME,
                    movement_unit="sec",
                    quality_score=0.2
                )

            top_leg = get_top_leg(l_hip, r_hip)
            foot_spread = abs(l_ankle[1] - r_ankle[1])
            is_active = foot_spread > 0.12

            if top_leg == "SOL":
                MAX_YAN_ACMA_SOL = max(MAX_YAN_ACMA_SOL, foot_spread)
                if is_active and not yan_completed_sol:
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
                MAX_YAN_ACMA_SAG = max(MAX_YAN_ACMA_SAG, foot_spread)
                if is_active and not yan_completed_sag:
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
                exercise_code="KALCA_YAN_ACMA",
                reps=reps,
                done=done,
                movement_name="side_lying_hip_abduction_hold",
                movement_value=max(MAX_YAN_ACMA_SOL, MAX_YAN_ACMA_SAG),
                movement_target=0.12,
                movement_unit="ratio",
                right_value=MAX_YAN_ACMA_SAG,
                left_value=MAX_YAN_ACMA_SOL,
                right_reps=1 if yan_completed_sag else 0,
                left_reps=1 if yan_completed_sol else 0,
                quality_score=0.75
            )

        # ==================== 5. YÜZÜSTÜ KALDIRMA ====================
        elif current_exercise == "KALCA_YUZUSTU":
            talimat = f"Yuzustu yat, bacagini geriye kaldir ({YUZUSTU_HOLD_TIME}sn her bacak)"

            if not check_prone(l_sh, l_hip):
                mesaj = "⚠️ Yüzüstü yatmalısın!"
                return talimat, mesaj, make_progress_payload(
                    exercise_code="KALCA_YUZUSTU",
                    reps=0,
                    done=False,
                    movement_name="prone_hip_extension_hold",
                    movement_value=0.0,
                    movement_target=YUZUSTU_HOLD_TIME,
                    movement_unit="sec",
                    quality_score=0.2
                )

            active_sol = l_ankle[1] < (l_hip[1] - 0.03)
            active_sag = r_ankle[1] < (r_hip[1] - 0.03)

            if active_sol and not yuzustu_completed_sol:
                if yuzustu_start_time_sol is None:
                    yuzustu_start_time_sol = time.time()
                elapsed = time.time() - yuzustu_start_time_sol
                if elapsed >= YUZUSTU_HOLD_TIME:
                    yuzustu_completed_sol = True
                    msg_sol = "✅ TAMAM!"
                else:
                    msg_sol = f"Tut! {int(YUZUSTU_HOLD_TIME - elapsed)}sn"
            elif yuzustu_completed_sol:
                msg_sol = "✅ TAMAM!"
            else:
                yuzustu_start_time_sol = None
                msg_sol = "Kaldır"

            if active_sag and not yuzustu_completed_sag:
                if yuzustu_start_time_sag is None:
                    yuzustu_start_time_sag = time.time()
                elapsed = time.time() - yuzustu_start_time_sag
                if elapsed >= YUZUSTU_HOLD_TIME:
                    yuzustu_completed_sag = True
                    msg_sag = "✅ TAMAM!"
                else:
                    msg_sag = f"Tut! {int(YUZUSTU_HOLD_TIME - elapsed)}sn"
            elif yuzustu_completed_sag:
                msg_sag = "✅ TAMAM!"
            else:
                yuzustu_start_time_sag = None
                msg_sag = "Kaldır"

            if active_sol:
                MAX_YUZUSTU_SOL = max(MAX_YUZUSTU_SOL, abs(l_hip[1] - l_ankle[1]))
            if active_sag:
                MAX_YUZUSTU_SAG = max(MAX_YUZUSTU_SAG, abs(r_hip[1] - r_ankle[1]))

            mesaj = f"Sol: {msg_sol} | Sağ: {msg_sag}"
            done = bool(yuzustu_completed_sol and yuzustu_completed_sag)
            reps = 1 if done else 0

            ekstra_bilgi = make_progress_payload(
                exercise_code="KALCA_YUZUSTU",
                reps=reps,
                done=done,
                movement_name="prone_hip_extension_hold",
                movement_value=max(MAX_YUZUSTU_SOL, MAX_YUZUSTU_SAG),
                movement_target=0.03,
                movement_unit="ratio",
                right_value=MAX_YUZUSTU_SAG,
                left_value=MAX_YUZUSTU_SOL,
                right_reps=1 if yuzustu_completed_sag else 0,
                left_reps=1 if yuzustu_completed_sol else 0,
                quality_score=0.76
            )

        # ==================== 6. YAN DİZ ÇEKME ====================
        elif current_exercise == "KALCA_YAN_DIZ_CEKME":
            talimat = "Yan yatarken dizini karnina cek (Her bacak 10'ar)"

            if not check_side_lying(l_sh, r_sh):
                mesaj = "⚠️ Yan yatmalısın!"
                return talimat, mesaj, make_progress_payload(
                    exercise_code="KALCA_YAN_DIZ_CEKME",
                    reps=0,
                    done=False,
                    movement_name="side_lying_hip_knee_flexion",
                    movement_value=0.0,
                    movement_target=DIZ_CEKME_THRESH,
                    movement_unit="deg",
                    quality_score=0.2
                )

            top_leg = get_top_leg(l_hip, r_hip)

            if top_leg == "SOL":
                angle = calculate_angle_3d(l_sh, l_hip, l_knee)
                MAX_YAN_DIZ_CEKME_SOL = max(MAX_YAN_DIZ_CEKME_SOL, angle)
                counter_yan_diz_sol.count(angle)
                sol_reps = int(counter_yan_diz_sol.rep_count)
                sag_reps = int(counter_yan_diz_sag.rep_count)
                done = sol_reps >= MAX_REPS and sag_reps >= MAX_REPS
                mesaj = f"SOL: {sol_reps}/10 ({int(angle)}°)"
            else:
                angle = calculate_angle_3d(r_sh, r_hip, r_knee)
                MAX_YAN_DIZ_CEKME_SAG = max(MAX_YAN_DIZ_CEKME_SAG, angle)
                counter_yan_diz_sag.count(angle)
                sol_reps = int(counter_yan_diz_sol.rep_count)
                sag_reps = int(counter_yan_diz_sag.rep_count)
                done = sol_reps >= MAX_REPS and sag_reps >= MAX_REPS
                mesaj = f"SAĞ: {sag_reps}/10 ({int(angle)}°)"

            reps = min_pair_reps(sol_reps, sag_reps)
            if done:
                mesaj = f"✅ TAMAMLANDI! Sol:{sol_reps}/10 Sağ:{sag_reps}/10"

            ekstra_bilgi = make_progress_payload(
                exercise_code="KALCA_YAN_DIZ_CEKME",
                reps=reps,
                done=done,
                movement_name="side_lying_hip_knee_flexion",
                movement_value=max(MAX_YAN_DIZ_CEKME_SOL, MAX_YAN_DIZ_CEKME_SAG),
                movement_target=DIZ_CEKME_THRESH,
                movement_unit="deg",
                right_value=MAX_YAN_DIZ_CEKME_SAG,
                left_value=MAX_YAN_DIZ_CEKME_SOL,
                right_reps=sag_reps,
                left_reps=sol_reps,
                quality_score=0.77
            )

        else:
            mesaj = "Bilinmeyen hareket"

    except Exception as e:
        mesaj = f"❌ Hata: {str(e)}"
        print(f"KALCA MODULU HATA: {e}")

    return talimat, mesaj, ekstra_bilgi