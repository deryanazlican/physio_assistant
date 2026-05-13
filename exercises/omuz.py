# exercises/omuz.py
# Progress-tracking uyumlu sürüm
# - Tüm egzersizler build_progress_payload ile standart veri döndürür
# - Sağ/sol max değerler tutulur
# - Simetri skoru üretilebilir
# - Timer bazlı egzersizler ortak formatla döner

import numpy as np
import mediapipe as mp
import time

from utils.counter import RepCounter
from utils.timer import DurationTimer
from utils.angles import calculate_angle_3d, calculate_distance_3d
from core.progress_metrics import build_progress_payload

mp_pose = mp.solutions.pose.PoseLandmark

# ==================== AYARLAR ====================
LEAN_THRESHOLD = 15
HIP_STABILITY_THRESHOLD = 0.03
CIRCLE_DURATION = 5

YANA_ACMA_THRESH = 60
ONE_ACMA_THRESH = 70
DISA_ACMA_THRESH = 40
DUVAR_YANA_THRESH = 60
DUVAR_ONE_THRESH = 65
DUVAR_GERIYE_THRESH = 30

MAX_REPS = 10

# ==================== GLOBAL STATE ====================
last_wrist_pos_sol = None
last_wrist_pos_sag = None
last_hip_x = None
last_hip_y = None

circle_start_time_sol = None
circle_start_time_sag = None
circle_completed_sol = False
circle_completed_sag = False

# Progress maxima
MAX_YANA_ACMA_SOL = 0.0
MAX_YANA_ACMA_SAG = 0.0

MAX_ONE_ACMA_SOL = 0.0
MAX_ONE_ACMA_SAG = 0.0

MAX_DISA_ACMA_SOL = 0.0
MAX_DISA_ACMA_SAG = 0.0

MAX_PEN_FLEKSIYON_SOL = 0.0
MAX_PEN_FLEKSIYON_SAG = 0.0

MAX_PEN_ABDUKSIYON_SOL = 0.0
MAX_PEN_ABDUKSIYON_SAG = 0.0

MAX_DUVAR_YANA_SOL = 0.0
MAX_DUVAR_YANA_SAG = 0.0

MAX_DUVAR_ONE_SOL = 0.0
MAX_DUVAR_ONE_SAG = 0.0

MAX_DUVAR_GERIYE_SOL = 0.0
MAX_DUVAR_GERIYE_SAG = 0.0

# ==================== SAYAÇLAR ====================
counter_yana_acma_sol = RepCounter("OMUZ_YANA_ACMA", "Sol", threshold_angle=YANA_ACMA_THRESH, target_reps=MAX_REPS, neutral_threshold=25)
counter_yana_acma_sag = RepCounter("OMUZ_YANA_ACMA", "Sag", threshold_angle=YANA_ACMA_THRESH, target_reps=MAX_REPS, neutral_threshold=25)

counter_one_acma_sol = RepCounter("OMUZ_ONE_ACMA", "Sol", threshold_angle=ONE_ACMA_THRESH, target_reps=MAX_REPS, neutral_threshold=30)
counter_one_acma_sag = RepCounter("OMUZ_ONE_ACMA", "Sag", threshold_angle=ONE_ACMA_THRESH, target_reps=MAX_REPS, neutral_threshold=30)

counter_disa_acma_sol = RepCounter("OMUZ_DISA_ACMA", "Sol", threshold_angle=DISA_ACMA_THRESH, target_reps=MAX_REPS, neutral_threshold=15)
counter_disa_acma_sag = RepCounter("OMUZ_DISA_ACMA", "Sag", threshold_angle=DISA_ACMA_THRESH, target_reps=MAX_REPS, neutral_threshold=15)

counter_pen_fleksiyon_sol = RepCounter("OMUZ_PEN_FLEKSIYON", "Sol", threshold_angle=30, target_reps=MAX_REPS, neutral_threshold=10)
counter_pen_fleksiyon_sag = RepCounter("OMUZ_PEN_FLEKSIYON", "Sag", threshold_angle=30, target_reps=MAX_REPS, neutral_threshold=10)

counter_pen_abduksiyon_sol = RepCounter("OMUZ_PEN_ABDUKSIYON", "Sol", threshold_angle=25, target_reps=MAX_REPS, neutral_threshold=5)
counter_pen_abduksiyon_sag = RepCounter("OMUZ_PEN_ABDUKSIYON", "Sag", threshold_angle=25, target_reps=MAX_REPS, neutral_threshold=5)

counter_duvar_yana_sol = RepCounter("OMUZ_DUVAR_YANA", "Sol", threshold_angle=DUVAR_YANA_THRESH, target_reps=MAX_REPS, neutral_threshold=30)
counter_duvar_yana_sag = RepCounter("OMUZ_DUVAR_YANA", "Sag", threshold_angle=DUVAR_YANA_THRESH, target_reps=MAX_REPS, neutral_threshold=30)

counter_duvar_one_sol = RepCounter("OMUZ_DUVAR_ONE", "Sol", threshold_angle=DUVAR_ONE_THRESH, target_reps=MAX_REPS, neutral_threshold=30)
counter_duvar_one_sag = RepCounter("OMUZ_DUVAR_ONE", "Sag", threshold_angle=DUVAR_ONE_THRESH, target_reps=MAX_REPS, neutral_threshold=30)

counter_duvar_geriye_sol = RepCounter("OMUZ_DUVAR_GERIYE", "Sol", threshold_angle=DUVAR_GERIYE_THRESH, target_reps=MAX_REPS, neutral_threshold=10)
counter_duvar_geriye_sag = RepCounter("OMUZ_DUVAR_GERIYE", "Sag", threshold_angle=DUVAR_GERIYE_THRESH, target_reps=MAX_REPS, neutral_threshold=10)

timer_germe_sol = DurationTimer("OMUZ_GERME", "Sol", target_duration=15)
timer_germe_sag = DurationTimer("OMUZ_GERME", "Sag", target_duration=15)

# ==================== RESET ====================
def reset_omuz_counters():
    global last_wrist_pos_sol, last_wrist_pos_sag, last_hip_x, last_hip_y
    global circle_start_time_sol, circle_start_time_sag, circle_completed_sol, circle_completed_sag

    global MAX_YANA_ACMA_SOL, MAX_YANA_ACMA_SAG
    global MAX_ONE_ACMA_SOL, MAX_ONE_ACMA_SAG
    global MAX_DISA_ACMA_SOL, MAX_DISA_ACMA_SAG
    global MAX_PEN_FLEKSIYON_SOL, MAX_PEN_FLEKSIYON_SAG
    global MAX_PEN_ABDUKSIYON_SOL, MAX_PEN_ABDUKSIYON_SAG
    global MAX_DUVAR_YANA_SOL, MAX_DUVAR_YANA_SAG
    global MAX_DUVAR_ONE_SOL, MAX_DUVAR_ONE_SAG
    global MAX_DUVAR_GERIYE_SOL, MAX_DUVAR_GERIYE_SAG

    counter_yana_acma_sol.reset(); counter_yana_acma_sag.reset()
    counter_one_acma_sol.reset(); counter_one_acma_sag.reset()
    counter_disa_acma_sol.reset(); counter_disa_acma_sag.reset()
    counter_pen_fleksiyon_sol.reset(); counter_pen_fleksiyon_sag.reset()
    counter_pen_abduksiyon_sol.reset(); counter_pen_abduksiyon_sag.reset()
    counter_duvar_yana_sol.reset(); counter_duvar_yana_sag.reset()
    counter_duvar_one_sol.reset(); counter_duvar_one_sag.reset()
    counter_duvar_geriye_sol.reset(); counter_duvar_geriye_sag.reset()
    timer_germe_sol.reset(); timer_germe_sag.reset()

    last_wrist_pos_sol = None
    last_wrist_pos_sag = None
    last_hip_x = None
    last_hip_y = None

    circle_start_time_sol = None
    circle_start_time_sag = None
    circle_completed_sol = False
    circle_completed_sag = False

    MAX_YANA_ACMA_SOL = 0.0
    MAX_YANA_ACMA_SAG = 0.0

    MAX_ONE_ACMA_SOL = 0.0
    MAX_ONE_ACMA_SAG = 0.0

    MAX_DISA_ACMA_SOL = 0.0
    MAX_DISA_ACMA_SAG = 0.0

    MAX_PEN_FLEKSIYON_SOL = 0.0
    MAX_PEN_FLEKSIYON_SAG = 0.0

    MAX_PEN_ABDUKSIYON_SOL = 0.0
    MAX_PEN_ABDUKSIYON_SAG = 0.0

    MAX_DUVAR_YANA_SOL = 0.0
    MAX_DUVAR_YANA_SAG = 0.0

    MAX_DUVAR_ONE_SOL = 0.0
    MAX_DUVAR_ONE_SAG = 0.0

    MAX_DUVAR_GERIYE_SOL = 0.0
    MAX_DUVAR_GERIYE_SAG = 0.0

    print("✅ Omuz modülü sıfırlandı.")

# ==================== YARDIMCI ====================
def check_visibility(landmarks, indices, threshold=0.3):
    for idx in indices:
        if landmarks[idx].visibility < threshold:
            return False
    return True


def check_stability(l_hip, r_hip):
    global last_hip_x, last_hip_y

    current_x = (l_hip[0] + r_hip[0]) / 2
    current_y = (l_hip[1] + r_hip[1]) / 2

    if last_hip_x is None:
        last_hip_x, last_hip_y = current_x, current_y
        return True

    diff_x = abs(current_x - last_hip_x)
    diff_y = abs(current_y - last_hip_y)
    last_hip_x, last_hip_y = current_x, current_y

    return not (diff_x > 0.012 or diff_y > 0.012)


def did_hip_lift(current_hip_y):
    global last_hip_y
    if last_hip_y is None:
        last_hip_y = current_hip_y
        return False

    diff = last_hip_y - current_hip_y
    last_hip_y = last_hip_y * 0.95 + current_hip_y * 0.05
    return diff > HIP_STABILITY_THRESHOLD


def calculate_trunk_lean(shoulder, hip):
    vertical_pt = [hip[0], hip[1] - 0.5, hip[2]]
    return calculate_angle_3d(shoulder, hip, vertical_pt)


def check_stick_hold(l_wrist, r_wrist):
    dist = calculate_distance_3d(l_wrist, r_wrist)
    return 0.05 <= dist <= 1.0


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

# ==================== ANA ====================
def get_exercise_feedback(current_exercise, landmarks):
    global last_wrist_pos_sol, last_wrist_pos_sag
    global circle_start_time_sol, circle_start_time_sag, circle_completed_sol, circle_completed_sag

    global MAX_YANA_ACMA_SOL, MAX_YANA_ACMA_SAG
    global MAX_ONE_ACMA_SOL, MAX_ONE_ACMA_SAG
    global MAX_DISA_ACMA_SOL, MAX_DISA_ACMA_SAG
    global MAX_PEN_FLEKSIYON_SOL, MAX_PEN_FLEKSIYON_SAG
    global MAX_PEN_ABDUKSIYON_SOL, MAX_PEN_ABDUKSIYON_SAG
    global MAX_DUVAR_YANA_SOL, MAX_DUVAR_YANA_SAG
    global MAX_DUVAR_ONE_SOL, MAX_DUVAR_ONE_SAG
    global MAX_DUVAR_GERIYE_SOL, MAX_DUVAR_GERIYE_SAG

    feedback_talimat = "Egzersiz secilmedi"
    feedback_mesaj = ""
    ekstra_bilgi = make_progress_payload(
        exercise_code=current_exercise,
        reps=0,
        done=False,
        movement_name="shoulder_motion",
        movement_value=0.0,
        movement_target=0.0,
        movement_unit="deg",
        quality_score=0.0
    )

    IDX_L_SH = mp_pose.LEFT_SHOULDER.value
    IDX_R_SH = mp_pose.RIGHT_SHOULDER.value
    IDX_L_EL = mp_pose.LEFT_ELBOW.value
    IDX_R_EL = mp_pose.RIGHT_ELBOW.value
    IDX_L_WR = mp_pose.LEFT_WRIST.value
    IDX_R_WR = mp_pose.RIGHT_WRIST.value
    IDX_L_HIP = mp_pose.LEFT_HIP.value
    IDX_R_HIP = mp_pose.RIGHT_HIP.value

    try:
        l_sh = [landmarks[IDX_L_SH].x, landmarks[IDX_L_SH].y, landmarks[IDX_L_SH].z]
        l_el = [landmarks[IDX_L_EL].x, landmarks[IDX_L_EL].y, landmarks[IDX_L_EL].z]
        l_wr = [landmarks[IDX_L_WR].x, landmarks[IDX_L_WR].y, landmarks[IDX_L_WR].z]
        l_hip = [landmarks[IDX_L_HIP].x, landmarks[IDX_L_HIP].y, landmarks[IDX_L_HIP].z]

        r_sh = [landmarks[IDX_R_SH].x, landmarks[IDX_R_SH].y, landmarks[IDX_R_SH].z]
        r_el = [landmarks[IDX_R_EL].x, landmarks[IDX_R_EL].y, landmarks[IDX_R_EL].z]
        r_wr = [landmarks[IDX_R_WR].x, landmarks[IDX_R_WR].y, landmarks[IDX_R_WR].z]
        r_hip = [landmarks[IDX_R_HIP].x, landmarks[IDX_R_HIP].y, landmarks[IDX_R_HIP].z]

        if not check_visibility(landmarks, [IDX_L_SH, IDX_R_SH]):
            return "⚠️ Omuzlar gorunmeli!", "Kameraya daha yakin gec", make_progress_payload(
                exercise_code=current_exercise,
                reps=0,
                done=False,
                movement_name="shoulder_motion",
                movement_value=0.0,
                movement_target=0.0,
                movement_unit="deg",
                quality_score=0.0
            )

        # ==================== SOPA EGZERSİZLERİ ====================
        if ("ACMA" in current_exercise) and ("DUVAR" not in current_exercise):
            if not check_visibility(landmarks, [IDX_L_WR, IDX_R_WR], threshold=0.2):
                return "Ellerini goster!", "⚠️ Bilekler gorunmuyor", make_progress_payload(
                    exercise_code=current_exercise,
                    reps=0,
                    done=False,
                    movement_name="shoulder_motion",
                    movement_value=0.0,
                    movement_target=0.0,
                    movement_unit="deg",
                    quality_score=0.0
                )

            if not check_stick_hold(l_wr, r_wr):
                return "Sopayi tut!", "Eller aralikli olmali", make_progress_payload(
                    exercise_code=current_exercise,
                    reps=0,
                    done=False,
                    movement_name="shoulder_motion",
                    movement_value=0.0,
                    movement_target=0.0,
                    movement_unit="deg",
                    quality_score=0.0
                )

            if not check_stability(l_hip, r_hip):
                return "Sabit dur", "Kalca cok oynuyor", make_progress_payload(
                    exercise_code=current_exercise,
                    reps=0,
                    done=False,
                    movement_name="shoulder_motion",
                    movement_value=0.0,
                    movement_target=0.0,
                    movement_unit="deg",
                    quality_score=0.0
                )

            # ---- YANA AÇMA ----
            if current_exercise == "OMUZ_YANA_ACMA":
                feedback_talimat = "1. Yana Acma: Sopayi yana kaldır (Her taraf 10'ar)"

                angle_sol = calculate_angle_3d(l_hip, l_sh, l_el)
                angle_sag = calculate_angle_3d(r_hip, r_sh, r_el)

                MAX_YANA_ACMA_SOL = max(MAX_YANA_ACMA_SOL, angle_sol)
                MAX_YANA_ACMA_SAG = max(MAX_YANA_ACMA_SAG, angle_sag)

                counter_yana_acma_sol.count(angle_sol)
                counter_yana_acma_sag.count(angle_sag)

                sol = int(counter_yana_acma_sol.rep_count)
                sag = int(counter_yana_acma_sag.rep_count)
                reps = min_pair_reps(sol, sag)
                done = (sol >= MAX_REPS and sag >= MAX_REPS)

                feedback_mesaj = f"Tekrar: {reps}/{MAX_REPS} | Sol:{sol}/10 Sağ:{sag}/10 | Açı:{int(angle_sol)}°/{int(angle_sag)}°"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_YANA_ACMA",
                    reps=reps,
                    done=done,
                    movement_name="shoulder_abduction_with_stick",
                    movement_value=(angle_sol + angle_sag) / 2,
                    movement_target=YANA_ACMA_THRESH,
                    movement_unit="deg",
                    right_value=MAX_YANA_ACMA_SAG,
                    left_value=MAX_YANA_ACMA_SOL,
                    right_reps=sag,
                    left_reps=sol,
                    quality_score=0.78
                )

                if done:
                    feedback_mesaj = f"✅ TAMAMLANDI! Tekrar: {MAX_REPS}/{MAX_REPS} | Sol:{sol} Sağ:{sag}"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

            # ---- ÖNE AÇMA ----
            if current_exercise == "OMUZ_ONE_ACMA":
                feedback_talimat = "3. One Acma: Sopayi one kaldır (Her taraf 10'ar)"

                hip_center_y = (l_hip[1] + r_hip[1]) / 2
                if did_hip_lift(hip_center_y):
                    return feedback_talimat, "⚠️ BEL KALKTI! Daha az yukari", make_progress_payload(
                        exercise_code="OMUZ_ONE_ACMA",
                        reps=min_pair_reps(counter_one_acma_sol.rep_count, counter_one_acma_sag.rep_count),
                        done=False,
                        movement_name="shoulder_flexion_with_stick",
                        movement_value=0.0,
                        movement_target=ONE_ACMA_THRESH,
                        movement_unit="deg",
                        quality_score=0.4
                    )

                angle_sol = calculate_angle_3d(l_hip, l_sh, l_el)
                angle_sag = calculate_angle_3d(r_hip, r_sh, r_el)

                MAX_ONE_ACMA_SOL = max(MAX_ONE_ACMA_SOL, angle_sol)
                MAX_ONE_ACMA_SAG = max(MAX_ONE_ACMA_SAG, angle_sag)

                counter_one_acma_sol.count(angle_sol)
                counter_one_acma_sag.count(angle_sag)

                sol = int(counter_one_acma_sol.rep_count)
                sag = int(counter_one_acma_sag.rep_count)
                reps = min_pair_reps(sol, sag)
                done = (sol >= MAX_REPS and sag >= MAX_REPS)

                feedback_mesaj = f"Tekrar: {reps}/{MAX_REPS} | Sol:{sol}/10 Sağ:{sag}/10 | Açı:{int(angle_sol)}°/{int(angle_sag)}°"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_ONE_ACMA",
                    reps=reps,
                    done=done,
                    movement_name="shoulder_flexion_with_stick",
                    movement_value=(angle_sol + angle_sag) / 2,
                    movement_target=ONE_ACMA_THRESH,
                    movement_unit="deg",
                    right_value=MAX_ONE_ACMA_SAG,
                    left_value=MAX_ONE_ACMA_SOL,
                    right_reps=sag,
                    left_reps=sol,
                    quality_score=0.8
                )

                if done:
                    feedback_mesaj = f"✅ TAMAMLANDI! Tekrar: {MAX_REPS}/{MAX_REPS} | Sol:{sol} Sağ:{sag}"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

            # ---- DIŞA AÇMA ----
            if current_exercise == "OMUZ_DISA_ACMA":
                feedback_talimat = "2. Disa Acma: Dirsek 90°, elleri disa ac (Her taraf 10'ar)"

                elbow_angle = calculate_angle_3d(l_sh, l_el, l_wr)
                if elbow_angle < 50 or elbow_angle > 140:
                    return feedback_talimat, f"⚠️ Dirsek 90° olsun (Şu an: {int(elbow_angle)}°)", make_progress_payload(
                        exercise_code="OMUZ_DISA_ACMA",
                        reps=min_pair_reps(counter_disa_acma_sol.rep_count, counter_disa_acma_sag.rep_count),
                        done=False,
                        movement_name="shoulder_external_rotation",
                        movement_value=0.0,
                        movement_target=DISA_ACMA_THRESH,
                        movement_unit="deg",
                        quality_score=0.4
                    )

                rot_sol = calculate_angle_3d(l_hip, l_sh, l_wr)
                rot_sag = calculate_angle_3d(r_hip, r_sh, r_wr)

                MAX_DISA_ACMA_SOL = max(MAX_DISA_ACMA_SOL, rot_sol)
                MAX_DISA_ACMA_SAG = max(MAX_DISA_ACMA_SAG, rot_sag)

                counter_disa_acma_sol.count(rot_sol)
                counter_disa_acma_sag.count(rot_sag)

                sol = int(counter_disa_acma_sol.rep_count)
                sag = int(counter_disa_acma_sag.rep_count)
                reps = min_pair_reps(sol, sag)
                done = (sol >= MAX_REPS and sag >= MAX_REPS)

                feedback_mesaj = f"Tekrar: {reps}/{MAX_REPS} | Sol:{sol}/10 Sağ:{sag}/10"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_DISA_ACMA",
                    reps=reps,
                    done=done,
                    movement_name="shoulder_external_rotation",
                    movement_value=(rot_sol + rot_sag) / 2,
                    movement_target=DISA_ACMA_THRESH,
                    movement_unit="deg",
                    right_value=MAX_DISA_ACMA_SAG,
                    left_value=MAX_DISA_ACMA_SOL,
                    right_reps=sag,
                    left_reps=sol,
                    quality_score=0.77
                )

                if done:
                    feedback_mesaj = f"✅ TAMAMLANDI! Tekrar: {MAX_REPS}/{MAX_REPS} | Sol:{sol} Sağ:{sag}"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

        # ==================== PENDUL ====================
        if ("PEN" in current_exercise) or (current_exercise == "OMUZ_CEMBER"):
            if not check_visibility(landmarks, [IDX_L_WR, IDX_R_WR], threshold=0.2):
                return "Kolunu goster!", "⚠️ Bilekler gorunmuyor", make_progress_payload(
                    exercise_code=current_exercise,
                    reps=0,
                    done=False,
                    movement_name="pendulum_motion",
                    movement_value=0.0,
                    movement_target=0.0,
                    movement_unit="deg",
                    quality_score=0.0
                )

            lean_angle_sol = calculate_trunk_lean(l_sh, l_hip)
            lean_angle_sag = calculate_trunk_lean(r_sh, r_hip)
            avg_lean = (lean_angle_sol + lean_angle_sag) / 2

            if avg_lean < LEAN_THRESHOLD:
                return "One egilin!", f"Eğilme: {int(avg_lean)}° (Hedef: >{LEAN_THRESHOLD}°)", make_progress_payload(
                    exercise_code=current_exercise,
                    reps=0,
                    done=False,
                    movement_name="trunk_lean",
                    movement_value=avg_lean,
                    movement_target=LEAN_THRESHOLD,
                    movement_unit="deg",
                    quality_score=0.3
                )

            # ---- ÖNDE SALLAMA ----
            if current_exercise == "OMUZ_PEN_FLEKSIYON":
                feedback_talimat = "4. Onde Sallama: One egik, one-arkaya salla (Her taraf 10'ar)"

                arm_angle_sol = calculate_angle_3d(l_hip, l_sh, l_wr)
                arm_angle_sag = calculate_angle_3d(r_hip, r_sh, r_wr)

                MAX_PEN_FLEKSIYON_SOL = max(MAX_PEN_FLEKSIYON_SOL, arm_angle_sol)
                MAX_PEN_FLEKSIYON_SAG = max(MAX_PEN_FLEKSIYON_SAG, arm_angle_sag)

                counter_pen_fleksiyon_sol.count(arm_angle_sol)
                counter_pen_fleksiyon_sag.count(arm_angle_sag)

                sol = int(counter_pen_fleksiyon_sol.rep_count)
                sag = int(counter_pen_fleksiyon_sag.rep_count)
                reps = min_pair_reps(sol, sag)
                done = (sol >= MAX_REPS and sag >= MAX_REPS)

                feedback_mesaj = f"Tekrar: {reps}/{MAX_REPS} | Sol:{sol}/10 Sağ:{sag}/10"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_PEN_FLEKSIYON",
                    reps=reps,
                    done=done,
                    movement_name="pendulum_flexion",
                    movement_value=(arm_angle_sol + arm_angle_sag) / 2,
                    movement_target=30,
                    movement_unit="deg",
                    right_value=MAX_PEN_FLEKSIYON_SAG,
                    left_value=MAX_PEN_FLEKSIYON_SOL,
                    right_reps=sag,
                    left_reps=sol,
                    quality_score=0.74
                )

                if done:
                    feedback_mesaj = f"✅ TAMAMLANDI! Tekrar: {MAX_REPS}/{MAX_REPS} | Sol:{sol} Sağ:{sag}"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

            # ---- YANDA SALLAMA ----
            if current_exercise == "OMUZ_PEN_ABDUKSIYON":
                feedback_talimat = "5. Yanda Sallama: Saga-sola salla (Her taraf 10'ar)"

                angle_sol = calculate_angle_3d(r_sh, l_sh, l_wr)
                angle_sag = calculate_angle_3d(l_sh, r_sh, r_wr)

                MAX_PEN_ABDUKSIYON_SOL = max(MAX_PEN_ABDUKSIYON_SOL, angle_sol)
                MAX_PEN_ABDUKSIYON_SAG = max(MAX_PEN_ABDUKSIYON_SAG, angle_sag)

                counter_pen_abduksiyon_sol.count(angle_sol)
                counter_pen_abduksiyon_sag.count(angle_sag)

                sol = int(counter_pen_abduksiyon_sol.rep_count)
                sag = int(counter_pen_abduksiyon_sag.rep_count)
                reps = min_pair_reps(sol, sag)
                done = (sol >= MAX_REPS and sag >= MAX_REPS)

                feedback_mesaj = f"Tekrar: {reps}/{MAX_REPS} | Sol:{sol}/10 Sağ:{sag}/10"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_PEN_ABDUKSIYON",
                    reps=reps,
                    done=done,
                    movement_name="pendulum_abduction",
                    movement_value=(angle_sol + angle_sag) / 2,
                    movement_target=25,
                    movement_unit="deg",
                    right_value=MAX_PEN_ABDUKSIYON_SAG,
                    left_value=MAX_PEN_ABDUKSIYON_SOL,
                    right_reps=sag,
                    left_reps=sol,
                    quality_score=0.74
                )

                if done:
                    feedback_mesaj = f"✅ TAMAMLANDI! Tekrar: {MAX_REPS}/{MAX_REPS} | Sol:{sol} Sağ:{sag}"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

            # ---- ÇEMBER (süre) ----
            if current_exercise == "OMUZ_CEMBER":
                feedback_talimat = f"6. Cember Cizme: One egik, {CIRCLE_DURATION}sn cember ciz"

                is_moving_sol = False
                is_moving_sag = False

                if last_wrist_pos_sol is not None:
                    if calculate_distance_3d(last_wrist_pos_sol, l_wr) > 0.005:
                        is_moving_sol = True
                last_wrist_pos_sol = l_wr

                if last_wrist_pos_sag is not None:
                    if calculate_distance_3d(last_wrist_pos_sag, r_wr) > 0.005:
                        is_moving_sag = True
                last_wrist_pos_sag = r_wr

                sol_timer = None
                if circle_completed_sol:
                    sol_msg = "✅ TAMAM!"
                    sol_timer = 0
                elif is_moving_sol:
                    if circle_start_time_sol is None:
                        circle_start_time_sol = time.time()
                    elapsed = time.time() - circle_start_time_sol
                    remaining = max(0.0, CIRCLE_DURATION - elapsed)
                    sol_timer = remaining
                    if elapsed >= CIRCLE_DURATION:
                        circle_completed_sol = True
                        sol_msg = "✅ TAMAM!"
                        sol_timer = 0
                    else:
                        sol_msg = f"Çiz... {int(remaining)}sn"
                else:
                    sol_msg = "Başla"
                    circle_start_time_sol = None
                    sol_timer = CIRCLE_DURATION

                sag_timer = None
                if circle_completed_sag:
                    sag_msg = "✅ TAMAM!"
                    sag_timer = 0
                elif is_moving_sag:
                    if circle_start_time_sag is None:
                        circle_start_time_sag = time.time()
                    elapsed = time.time() - circle_start_time_sag
                    remaining = max(0.0, CIRCLE_DURATION - elapsed)
                    sag_timer = remaining
                    if elapsed >= CIRCLE_DURATION:
                        circle_completed_sag = True
                        sag_msg = "✅ TAMAM!"
                        sag_timer = 0
                    else:
                        sag_msg = f"Çiz... {int(remaining)}sn"
                else:
                    sag_msg = "Başla"
                    circle_start_time_sag = None
                    sag_timer = CIRCLE_DURATION

                done = bool(circle_completed_sol and circle_completed_sag)
                reps = 1 if done else 0

                feedback_mesaj = f"Sol: {sol_msg} | Sağ: {sag_msg}"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_CEMBER",
                    reps=reps,
                    done=done,
                    movement_name="shoulder_circle_duration",
                    movement_value=0.0,
                    movement_target=CIRCLE_DURATION,
                    movement_unit="sec",
                    right_value=float(CIRCLE_DURATION if circle_completed_sag else max(0.0, CIRCLE_DURATION - (sag_timer or CIRCLE_DURATION))),
                    left_value=float(CIRCLE_DURATION if circle_completed_sol else max(0.0, CIRCLE_DURATION - (sol_timer or CIRCLE_DURATION))),
                    right_reps=1 if circle_completed_sag else 0,
                    left_reps=1 if circle_completed_sol else 0,
                    quality_score=0.72,
                    timer=min(sol_timer, sag_timer)
                )

                if done:
                    feedback_mesaj = "✅ TAMAMLANDI! Tekrar: 1/1"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

        # ==================== DUVAR / GERME ====================
        if ("DUVAR" in current_exercise) or (current_exercise == "OMUZ_GERME"):
            if not check_stability(l_hip, r_hip):
                return "Sabit dur", "Kalca cok oynuyor", make_progress_payload(
                    exercise_code=current_exercise,
                    reps=0,
                    done=False,
                    movement_name="shoulder_motion",
                    movement_value=0.0,
                    movement_target=0.0,
                    movement_unit="deg",
                    quality_score=0.0
                )

            if not check_visibility(landmarks, [IDX_L_EL, IDX_R_EL], threshold=0.2):
                return "Kolunu goster!", "⚠️ Dirsekler gorunmuyor", make_progress_payload(
                    exercise_code=current_exercise,
                    reps=0,
                    done=False,
                    movement_name="shoulder_motion",
                    movement_value=0.0,
                    movement_target=0.0,
                    movement_unit="deg",
                    quality_score=0.0
                )

            # ---- DUVARA YANA ----
            if current_exercise == "OMUZ_DUVAR_YANA":
                feedback_talimat = "7. Duvara Yana: YAN dur, kolunu yana ac (Her taraf 10'ar)"

                angle_sol = calculate_angle_3d(l_hip, l_sh, l_el)
                angle_sag = calculate_angle_3d(r_hip, r_sh, r_el)

                MAX_DUVAR_YANA_SOL = max(MAX_DUVAR_YANA_SOL, angle_sol)
                MAX_DUVAR_YANA_SAG = max(MAX_DUVAR_YANA_SAG, angle_sag)

                counter_duvar_yana_sol.count(angle_sol)
                counter_duvar_yana_sag.count(angle_sag)

                sol = int(counter_duvar_yana_sol.rep_count)
                sag = int(counter_duvar_yana_sag.rep_count)
                reps = min_pair_reps(sol, sag)
                done = (sol >= MAX_REPS and sag >= MAX_REPS)

                feedback_mesaj = f"Tekrar: {reps}/{MAX_REPS} | Sol:{sol}/10 Sağ:{sag}/10"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_DUVAR_YANA",
                    reps=reps,
                    done=done,
                    movement_name="wall_abduction",
                    movement_value=(angle_sol + angle_sag) / 2,
                    movement_target=DUVAR_YANA_THRESH,
                    movement_unit="deg",
                    right_value=MAX_DUVAR_YANA_SAG,
                    left_value=MAX_DUVAR_YANA_SOL,
                    right_reps=sag,
                    left_reps=sol,
                    quality_score=0.78
                )

                if done:
                    feedback_mesaj = f"✅ TAMAMLANDI! Tekrar: {MAX_REPS}/{MAX_REPS} | Sol:{sol} Sağ:{sag}"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

            # ---- DUVARA ÖNE ----
            if current_exercise == "OMUZ_DUVAR_ONE":
                feedback_talimat = "8. Duvara One: Kolunu one it (Her taraf 10'ar)"

                angle_sol = calculate_angle_3d(l_hip, l_sh, l_el)
                angle_sag = calculate_angle_3d(r_hip, r_sh, r_el)

                MAX_DUVAR_ONE_SOL = max(MAX_DUVAR_ONE_SOL, angle_sol)
                MAX_DUVAR_ONE_SAG = max(MAX_DUVAR_ONE_SAG, angle_sag)

                counter_duvar_one_sol.count(angle_sol)
                counter_duvar_one_sag.count(angle_sag)

                sol = int(counter_duvar_one_sol.rep_count)
                sag = int(counter_duvar_one_sag.rep_count)
                reps = min_pair_reps(sol, sag)
                done = (sol >= MAX_REPS and sag >= MAX_REPS)

                feedback_mesaj = f"Tekrar: {reps}/{MAX_REPS} | Sol:{sol}/10 Sağ:{sag}/10"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_DUVAR_ONE",
                    reps=reps,
                    done=done,
                    movement_name="wall_flexion",
                    movement_value=(angle_sol + angle_sag) / 2,
                    movement_target=DUVAR_ONE_THRESH,
                    movement_unit="deg",
                    right_value=MAX_DUVAR_ONE_SAG,
                    left_value=MAX_DUVAR_ONE_SOL,
                    right_reps=sag,
                    left_reps=sol,
                    quality_score=0.79
                )

                if done:
                    feedback_mesaj = f"✅ TAMAMLANDI! Tekrar: {MAX_REPS}/{MAX_REPS} | Sol:{sol} Sağ:{sag}"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

            # ---- DUVARA GERİYE ----
            if current_exercise == "OMUZ_DUVAR_GERIYE":
                feedback_talimat = "9. Duvara Geriye: Kolunu geriye it (Her taraf 10'ar)"

                angle_sol = calculate_angle_3d(l_hip, l_sh, l_wr)
                angle_sag = calculate_angle_3d(r_hip, r_sh, r_wr)

                MAX_DUVAR_GERIYE_SOL = max(MAX_DUVAR_GERIYE_SOL, angle_sol)
                MAX_DUVAR_GERIYE_SAG = max(MAX_DUVAR_GERIYE_SAG, angle_sag)

                counter_duvar_geriye_sol.count(angle_sol)
                counter_duvar_geriye_sag.count(angle_sag)

                sol = int(counter_duvar_geriye_sol.rep_count)
                sag = int(counter_duvar_geriye_sag.rep_count)
                reps = min_pair_reps(sol, sag)
                done = (sol >= MAX_REPS and sag >= MAX_REPS)

                feedback_mesaj = f"Tekrar: {reps}/{MAX_REPS} | Sol:{sol}/10 Sağ:{sag}/10"
                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_DUVAR_GERIYE",
                    reps=reps,
                    done=done,
                    movement_name="wall_extension",
                    movement_value=(angle_sol + angle_sag) / 2,
                    movement_target=DUVAR_GERIYE_THRESH,
                    movement_unit="deg",
                    right_value=MAX_DUVAR_GERIYE_SAG,
                    left_value=MAX_DUVAR_GERIYE_SOL,
                    right_reps=sag,
                    left_reps=sol,
                    quality_score=0.77
                )

                if done:
                    feedback_mesaj = f"✅ TAMAMLANDI! Tekrar: {MAX_REPS}/{MAX_REPS} | Sol:{sol} Sağ:{sag}"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

            # ---- GERME (timer) ----
            if current_exercise == "OMUZ_GERME":
                feedback_talimat = "10. Germe (15sn): Kolunu gogse cek ve tut"

                active_sol = calculate_angle_3d(l_el, l_sh, r_sh) < 50
                active_sag = calculate_angle_3d(r_el, r_sh, l_sh) < 50

                msg_sol = timer_germe_sol.update_feedback(active_sol)
                msg_sag = timer_germe_sag.update_feedback(active_sag)

                done = bool(getattr(timer_germe_sol, "completed", False) and getattr(timer_germe_sag, "completed", False))
                feedback_mesaj = f"Sol: {msg_sol} | Sağ: {msg_sag}"

                elapsed_sol = 15.0 if getattr(timer_germe_sol, "completed", False) else max(0.0, getattr(timer_germe_sol, "elapsed_time", 0.0))
                elapsed_sag = 15.0 if getattr(timer_germe_sag, "completed", False) else max(0.0, getattr(timer_germe_sag, "elapsed_time", 0.0))

                ekstra_bilgi = make_progress_payload(
                    exercise_code="OMUZ_GERME",
                    reps=1 if done else 0,
                    done=done,
                    movement_name="shoulder_stretch_hold",
                    movement_value=(elapsed_sol + elapsed_sag) / 2.0,
                    movement_target=15.0,
                    movement_unit="sec",
                    right_value=elapsed_sag,
                    left_value=elapsed_sol,
                    right_reps=1 if getattr(timer_germe_sag, "completed", False) else 0,
                    left_reps=1 if getattr(timer_germe_sol, "completed", False) else 0,
                    quality_score=0.76
                )

                if done:
                    feedback_mesaj = "✅ TAMAMLANDI! Tekrar: 1/1"

                return feedback_talimat, feedback_mesaj, ekstra_bilgi

        return feedback_talimat, "⚠️ Uygun pozisyon al", make_progress_payload(
            exercise_code=current_exercise,
            reps=0,
            done=False,
            movement_name="shoulder_motion",
            movement_value=0.0,
            movement_target=0.0,
            movement_unit="deg",
            quality_score=0.0
        )

    except Exception as e:
        print(f"OMUZ MODULU HATA: {e}")
        return "❌ Hata", f"{str(e)}", make_progress_payload(
            exercise_code=current_exercise,
            reps=0,
            done=False,
            movement_name="shoulder_motion",
            movement_value=0.0,
            movement_target=0.0,
            movement_unit="deg",
            quality_score=0.0
        )