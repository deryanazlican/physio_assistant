import mediapipe as mp
import time
import numpy as np

from utils.counter import RepCounter
from utils.angles import calculate_angle_3d
from core.progress_metrics import build_progress_payload

mp_pose = mp.solutions.pose.PoseLandmark

# ==================== AYARLAR ====================
TEK_DIZ_THRESH = 70
CIFT_DIZ_THRESH = 70
MEKIK_THRESH = 120
SLR_THRESH = 35
KOPRU_THRESH = 150
KEDI_THRESH = 15
YUZUSTU_HOLD_TIME = 6
KOPRU_HOLD_TIME = 5

MAX_REPS = 10

# ==================== SAYAÇLAR ====================
counter_tek_diz_sol = RepCounter("BEL_TEK_DIZ", "Sol", threshold_angle=TEK_DIZ_THRESH, target_reps=MAX_REPS, neutral_threshold=140)
counter_tek_diz_sag = RepCounter("BEL_TEK_DIZ", "Sag", threshold_angle=TEK_DIZ_THRESH, target_reps=MAX_REPS, neutral_threshold=140)

counter_cift_diz = RepCounter("BEL_CIFT_DIZ", "Cift", threshold_angle=CIFT_DIZ_THRESH, target_reps=MAX_REPS, neutral_threshold=140)

counter_mekik = RepCounter("BEL_MEKIK", "Karin", threshold_angle=MEKIK_THRESH, target_reps=MAX_REPS, neutral_threshold=160)

counter_slr_sol = RepCounter("BEL_SLR", "Sol", threshold_angle=SLR_THRESH, target_reps=MAX_REPS, neutral_threshold=10)
counter_slr_sag = RepCounter("BEL_SLR", "Sag", threshold_angle=SLR_THRESH, target_reps=MAX_REPS, neutral_threshold=10)

counter_kedi = RepCounter("BEL_KEDI_DEVE", "Omurga", threshold_angle=KEDI_THRESH, target_reps=MAX_REPS, neutral_threshold=5)

# ==================== GLOBAL STATE ====================
kopru_start_time = None
kopru_completed = False

yuzustu_start_time = None
yuzustu_completed = False

# Progress maxima
MAX_TEK_DIZ_SOL = 0.0
MAX_TEK_DIZ_SAG = 0.0

MAX_CIFT_DIZ = 0.0

MAX_MEKIK = 0.0

MAX_SLR_SOL = 0.0
MAX_SLR_SAG = 0.0

MAX_KOPRU = 0.0

MAX_KEDI = 0.0

MAX_YUZUSTU = 0.0

# ==================== RESET ====================
def reset_bel_counters():
    global kopru_start_time, kopru_completed, yuzustu_start_time, yuzustu_completed
    global MAX_TEK_DIZ_SOL, MAX_TEK_DIZ_SAG
    global MAX_CIFT_DIZ
    global MAX_MEKIK
    global MAX_SLR_SOL, MAX_SLR_SAG
    global MAX_KOPRU
    global MAX_KEDI
    global MAX_YUZUSTU

    counter_tek_diz_sol.reset()
    counter_tek_diz_sag.reset()
    counter_cift_diz.reset()
    counter_mekik.reset()
    counter_slr_sol.reset()
    counter_slr_sag.reset()
    counter_kedi.reset()

    kopru_start_time = None
    kopru_completed = False
    yuzustu_start_time = None
    yuzustu_completed = False

    MAX_TEK_DIZ_SOL = 0.0
    MAX_TEK_DIZ_SAG = 0.0
    MAX_CIFT_DIZ = 0.0
    MAX_MEKIK = 0.0
    MAX_SLR_SOL = 0.0
    MAX_SLR_SAG = 0.0
    MAX_KOPRU = 0.0
    MAX_KEDI = 0.0
    MAX_YUZUSTU = 0.0

    print("✅ Bel modülü sıfırlandı.")

# ==================== YARDIMCI FONKSİYONLAR ====================
def get_lm(landmarks, lm_name):
    lm = landmarks[lm_name.value]
    if lm.visibility < 0.25:
        return None
    z_val = getattr(lm, "z", 0.0)
    return [lm.x, lm.y, z_val]


def avg(a, b):
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]


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
    global kopru_start_time, kopru_completed, yuzustu_start_time, yuzustu_completed
    global MAX_TEK_DIZ_SOL, MAX_TEK_DIZ_SAG
    global MAX_CIFT_DIZ
    global MAX_MEKIK
    global MAX_SLR_SOL, MAX_SLR_SAG
    global MAX_KOPRU
    global MAX_KEDI
    global MAX_YUZUSTU

    talimat = ""
    mesaj = ""
    ekstra_bilgi = make_progress_payload(
        exercise_code=current_exercise,
        reps=0,
        done=False,
        movement_name="lumbar_motion",
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
        nose = get_lm(landmarks, mp_pose.NOSE)

        if not l_hip or not r_hip or not l_knee or not r_knee:
            return "⚠️ Gorunmuyorsun", "Kameraya gec", make_progress_payload(
                exercise_code=current_exercise,
                reps=0,
                done=False,
                movement_name="lumbar_motion",
                movement_value=0.0,
                movement_target=0.0,
                movement_unit="deg",
                quality_score=0.0
            )

        # ==================== 1. TEK DİZ ÇEKME ====================
        if current_exercise == "BEL_TEK_DIZ":
            talimat = "Tek dizi gogsune cek (Her bacak 10'ar)"

            angle_l = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_r = calculate_angle_3d(r_sh, r_hip, r_knee)

            flex_l = 180 - angle_l
            flex_r = 180 - angle_r

            MAX_TEK_DIZ_SOL = max(MAX_TEK_DIZ_SOL, flex_l)
            MAX_TEK_DIZ_SAG = max(MAX_TEK_DIZ_SAG, flex_r)

            sol_reps = int(counter_tek_diz_sol.rep_count)
            sag_reps = int(counter_tek_diz_sag.rep_count)

            if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_TEK_DIZ",
                    reps=MAX_REPS,
                    done=True,
                    movement_name="single_knee_to_chest",
                    movement_value=max(MAX_TEK_DIZ_SOL, MAX_TEK_DIZ_SAG),
                    movement_target=TEK_DIZ_THRESH,
                    movement_unit="deg",
                    right_value=MAX_TEK_DIZ_SAG,
                    left_value=MAX_TEK_DIZ_SOL,
                    right_reps=sag_reps,
                    left_reps=sol_reps,
                    quality_score=0.8
                )
            else:
                if flex_l > 30 and flex_r < 20:
                    counter_tek_diz_sol.count(flex_l)
                    sol_reps = int(counter_tek_diz_sol.rep_count)
                    mesaj = f"SOL: {sol_reps}/10 ({int(flex_l)}°)"
                elif flex_r > 30 and flex_l < 20:
                    counter_tek_diz_sag.count(flex_r)
                    sag_reps = int(counter_tek_diz_sag.rep_count)
                    mesaj = f"SAĞ: {sag_reps}/10 ({int(flex_r)}°)"
                else:
                    counter_tek_diz_sol.count(0)
                    counter_tek_diz_sag.count(0)
                    mesaj = "Tek bacak bukulmeli!"

                sol_reps = int(counter_tek_diz_sol.rep_count)
                sag_reps = int(counter_tek_diz_sag.rep_count)

                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_TEK_DIZ",
                    reps=min(sol_reps, sag_reps),
                    done=False,
                    movement_name="single_knee_to_chest",
                    movement_value=max(flex_l, flex_r),
                    movement_target=TEK_DIZ_THRESH,
                    movement_unit="deg",
                    right_value=MAX_TEK_DIZ_SAG,
                    left_value=MAX_TEK_DIZ_SOL,
                    right_reps=sag_reps,
                    left_reps=sol_reps,
                    quality_score=0.75
                )

        # ==================== 2. ÇİFT DİZ ÇEKME ====================
        elif current_exercise == "BEL_CIFT_DIZ":
            talimat = "Iki dizi birden gogsune cek (10 tekrar)"

            angle_l = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_r = calculate_angle_3d(r_sh, r_hip, r_knee)
            avg_angle = (angle_l + angle_r) / 2
            flex = 180 - avg_angle

            MAX_CIFT_DIZ = max(MAX_CIFT_DIZ, flex)

            counter_cift_diz.count(flex)
            reps = int(counter_cift_diz.rep_count)
            done = reps >= MAX_REPS

            if done:
                mesaj = f"✅ TAMAMLANDI! ({MAX_REPS} tekrar)"
            else:
                mesaj = f"{reps}/10 | {int(flex)}°"

            ekstra_bilgi = make_progress_payload(
                exercise_code="BEL_CIFT_DIZ",
                reps=reps,
                done=done,
                movement_name="double_knee_to_chest",
                movement_value=flex,
                movement_target=CIFT_DIZ_THRESH,
                movement_unit="deg",
                right_value=flex,
                left_value=flex,
                right_reps=reps,
                left_reps=reps,
                quality_score=0.79
            )

        # ==================== 3. YARIM MEKİK ====================
        elif current_exercise == "BEL_MEKIK":
            talimat = "Omuzlarini kaldir, dizlere uzan (10 tekrar)"

            shoulder_mid = avg(l_sh, r_sh)
            hip_mid = avg(l_hip, r_hip)

            torso_vec = np.array([shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]])
            vert = np.array([0.0, -1.0])
            norm = np.linalg.norm(torso_vec)

            if norm == 0:
                torso_angle = 0.0
            else:
                dot = np.dot(torso_vec, vert) / norm
                torso_angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))

            flex_amount = 90 - torso_angle
            MAX_MEKIK = max(MAX_MEKIK, flex_amount)

            reps = int(counter_mekik.rep_count)

            if reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! ({MAX_REPS} tekrar)"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_MEKIK",
                    reps=MAX_REPS,
                    done=True,
                    movement_name="partial_crunch",
                    movement_value=MAX_MEKIK,
                    movement_target=MEKIK_THRESH,
                    movement_unit="deg",
                    right_value=MAX_MEKIK,
                    left_value=MAX_MEKIK,
                    right_reps=MAX_REPS,
                    left_reps=MAX_REPS,
                    quality_score=0.78
                )
            else:
                if flex_amount > 10:
                    counter_mekik.count(flex_amount)
                    reps = int(counter_mekik.rep_count)
                    mesaj = f"{reps}/10 | Fleksiyon: {int(flex_amount)}°"
                else:
                    counter_mekik.count(0)
                    mesaj = f"Daha fazla kaldir ({int(flex_amount)}°)"

                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_MEKIK",
                    reps=reps,
                    done=False,
                    movement_name="partial_crunch",
                    movement_value=flex_amount,
                    movement_target=MEKIK_THRESH,
                    movement_unit="deg",
                    right_value=MAX_MEKIK,
                    left_value=MAX_MEKIK,
                    right_reps=reps,
                    left_reps=reps,
                    quality_score=0.74
                )

        # ==================== 4. DÜZ BACAK KALDIRMA (SLR) ====================
        elif current_exercise == "BEL_SLR":
            talimat = "Diz bukmeden bacagi kaldir (Her bacak 10'ar)"

            knee_l = calculate_angle_3d(l_hip, l_knee, l_ankle)
            knee_r = calculate_angle_3d(r_hip, r_knee, r_ankle)

            hip_angle_l = calculate_angle_3d(l_sh, l_hip, l_knee)
            hip_angle_r = calculate_angle_3d(r_sh, r_hip, r_knee)

            flex_l = 180 - hip_angle_l
            flex_r = 180 - hip_angle_r

            MAX_SLR_SOL = max(MAX_SLR_SOL, flex_l)
            MAX_SLR_SAG = max(MAX_SLR_SAG, flex_r)

            sol_reps = int(counter_slr_sol.rep_count)
            sag_reps = int(counter_slr_sag.rep_count)

            if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_SLR",
                    reps=MAX_REPS,
                    done=True,
                    movement_name="straight_leg_raise",
                    movement_value=max(MAX_SLR_SOL, MAX_SLR_SAG),
                    movement_target=SLR_THRESH,
                    movement_unit="deg",
                    right_value=MAX_SLR_SAG,
                    left_value=MAX_SLR_SOL,
                    right_reps=sag_reps,
                    left_reps=sol_reps,
                    quality_score=0.8
                )
            else:
                if knee_r > 150 and flex_l > 25 and not (flex_r > 25):
                    if knee_l < 140:
                        mesaj = "SOL DİZİ DÜZELT!"
                    else:
                        counter_slr_sol.count(flex_l)
                        sol_reps = int(counter_slr_sol.rep_count)
                        mesaj = f"SOL: {sol_reps}/10 | {int(flex_l)}°"
                elif knee_l > 150 and flex_r > 25 and not (flex_l > 25):
                    if knee_r < 140:
                        mesaj = "SAĞ DİZİ DÜZELT!"
                    else:
                        counter_slr_sag.count(flex_r)
                        sag_reps = int(counter_slr_sag.rep_count)
                        mesaj = f"SAĞ: {sag_reps}/10 | {int(flex_r)}°"
                else:
                    counter_slr_sol.count(0)
                    counter_slr_sag.count(0)
                    mesaj = "Tek bacak kaldirmali!"

                sol_reps = int(counter_slr_sol.rep_count)
                sag_reps = int(counter_slr_sag.rep_count)

                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_SLR",
                    reps=min(sol_reps, sag_reps),
                    done=False,
                    movement_name="straight_leg_raise",
                    movement_value=max(flex_l, flex_r),
                    movement_target=SLR_THRESH,
                    movement_unit="deg",
                    right_value=MAX_SLR_SAG,
                    left_value=MAX_SLR_SOL,
                    right_reps=sag_reps,
                    left_reps=sol_reps,
                    quality_score=0.76,
                    extra={
                        "left_knee_extension_angle": float(knee_l),
                        "right_knee_extension_angle": float(knee_r),
                    }
                )

        # ==================== 5. KÖPRÜ ====================
        elif current_exercise == "BEL_KOPRU":
            talimat = f"Kalcani kaldir ve tut ({KOPRU_HOLD_TIME}sn)"

            angle_l = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_r = calculate_angle_3d(r_sh, r_hip, r_knee)
            avg_angle = (angle_l + angle_r) / 2
            MAX_KOPRU = max(MAX_KOPRU, avg_angle)

            is_up = avg_angle > KOPRU_THRESH

            if is_up and not kopru_completed:
                if kopru_start_time is None:
                    kopru_start_time = time.time()
                elapsed = time.time() - kopru_start_time
                if elapsed >= KOPRU_HOLD_TIME:
                    kopru_completed = True
                    mesaj = "✅ HARIKA TAMAMLANDI!"
                    ekstra_bilgi = make_progress_payload(
                        exercise_code="BEL_KOPRU",
                        reps=1,
                        done=True,
                        movement_name="bridge_hold",
                        movement_value=avg_angle,
                        movement_target=KOPRU_THRESH,
                        movement_unit="deg",
                        right_value=angle_r,
                        left_value=angle_l,
                        right_reps=1,
                        left_reps=1,
                        quality_score=0.82,
                        timer=0
                    )
                else:
                    remaining = KOPRU_HOLD_TIME - elapsed
                    mesaj = f"💪 TUT! {int(remaining)}sn | {int(avg_angle)}°"
                    ekstra_bilgi = make_progress_payload(
                        exercise_code="BEL_KOPRU",
                        reps=0,
                        done=False,
                        movement_name="bridge_hold",
                        movement_value=avg_angle,
                        movement_target=KOPRU_THRESH,
                        movement_unit="deg",
                        right_value=angle_r,
                        left_value=angle_l,
                        right_reps=0,
                        left_reps=0,
                        quality_score=0.78,
                        timer=remaining
                    )
            elif kopru_completed:
                mesaj = "✅ TAMAMLANDI!"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_KOPRU",
                    reps=1,
                    done=True,
                    movement_name="bridge_hold",
                    movement_value=MAX_KOPRU,
                    movement_target=KOPRU_THRESH,
                    movement_unit="deg",
                    right_value=MAX_KOPRU,
                    left_value=MAX_KOPRU,
                    right_reps=1,
                    left_reps=1,
                    quality_score=0.82,
                    timer=0
                )
            else:
                kopru_start_time = None
                mesaj = f"Daha fazla kaldir ({int(avg_angle)}°)"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_KOPRU",
                    reps=0,
                    done=False,
                    movement_name="bridge_hold",
                    movement_value=avg_angle,
                    movement_target=KOPRU_THRESH,
                    movement_unit="deg",
                    right_value=angle_r,
                    left_value=angle_l,
                    quality_score=0.55
                )

        # ==================== 6. KEDİ - DEVE ====================
        elif current_exercise == "BEL_KEDI_DEVE":
            talimat = "Sirtini kambur yap - duz yap (10 tekrar)"

            shoulder_mid = avg(l_sh, r_sh)
            hip_mid = avg(l_hip, r_hip)
            spine_diff = abs((shoulder_mid[1] - hip_mid[1]) * 80)

            MAX_KEDI = max(MAX_KEDI, spine_diff)

            counter_kedi.count(spine_diff)
            reps = int(counter_kedi.rep_count)
            done = reps >= MAX_REPS

            if done:
                mesaj = f"✅ TAMAMLANDI! ({MAX_REPS} tekrar)"
            else:
                mesaj = f"{reps}/10 | Hareket: {int(spine_diff)}"

            ekstra_bilgi = make_progress_payload(
                exercise_code="BEL_KEDI_DEVE",
                reps=reps,
                done=done,
                movement_name="cat_camel_spinal_motion",
                movement_value=spine_diff,
                movement_target=KEDI_THRESH,
                movement_unit="ratio",
                right_value=MAX_KEDI,
                left_value=MAX_KEDI,
                right_reps=reps,
                left_reps=reps,
                quality_score=0.77
            )

        # ==================== 7. YÜZÜSTÜ DOĞRULMA ====================
        elif current_exercise == "BEL_YUZUSTU":
            talimat = f"Yuzustu yat, govdeni kaldir ({YUZUSTU_HOLD_TIME}sn)"

            lift_value = max(
                (l_hip[1] - l_sh[1]),
                (r_hip[1] - r_sh[1])
            )
            MAX_YUZUSTU = max(MAX_YUZUSTU, lift_value)

            is_lifted = (l_sh[1] < (l_hip[1] - 0.06)) and (r_sh[1] < (r_hip[1] - 0.06))

            if is_lifted and not yuzustu_completed:
                if yuzustu_start_time is None:
                    yuzustu_start_time = time.time()
                elapsed = time.time() - yuzustu_start_time
                if elapsed >= YUZUSTU_HOLD_TIME:
                    yuzustu_completed = True
                    mesaj = "✅ HARIKA TAMAMLANDI!"
                    ekstra_bilgi = make_progress_payload(
                        exercise_code="BEL_YUZUSTU",
                        reps=1,
                        done=True,
                        movement_name="prone_trunk_extension_hold",
                        movement_value=lift_value,
                        movement_target=0.06,
                        movement_unit="ratio",
                        right_value=lift_value,
                        left_value=lift_value,
                        right_reps=1,
                        left_reps=1,
                        quality_score=0.8,
                        timer=0
                    )
                else:
                    remaining = YUZUSTU_HOLD_TIME - elapsed
                    mesaj = f"💪 TUT! {int(remaining)}sn"
                    ekstra_bilgi = make_progress_payload(
                        exercise_code="BEL_YUZUSTU",
                        reps=0,
                        done=False,
                        movement_name="prone_trunk_extension_hold",
                        movement_value=lift_value,
                        movement_target=0.06,
                        movement_unit="ratio",
                        right_value=lift_value,
                        left_value=lift_value,
                        right_reps=0,
                        left_reps=0,
                        quality_score=0.76,
                        timer=remaining
                    )
            elif yuzustu_completed:
                mesaj = "✅ TAMAMLANDI!"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_YUZUSTU",
                    reps=1,
                    done=True,
                    movement_name="prone_trunk_extension_hold",
                    movement_value=MAX_YUZUSTU,
                    movement_target=0.06,
                    movement_unit="ratio",
                    right_value=MAX_YUZUSTU,
                    left_value=MAX_YUZUSTU,
                    right_reps=1,
                    left_reps=1,
                    quality_score=0.8,
                    timer=0
                )
            else:
                yuzustu_start_time = None
                mesaj = "Daha fazla kaldir"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="BEL_YUZUSTU",
                    reps=0,
                    done=False,
                    movement_name="prone_trunk_extension_hold",
                    movement_value=lift_value,
                    movement_target=0.06,
                    movement_unit="ratio",
                    right_value=lift_value,
                    left_value=lift_value,
                    quality_score=0.5
                )

        else:
            mesaj = "Bilinmeyen hareket"

    except Exception as e:
        mesaj = f"❌ Hata: {str(e)}"
        print(f"BEL MODULU HATA: {e}")

    return talimat, mesaj, ekstra_bilgi