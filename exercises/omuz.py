import numpy as np
import mediapipe as mp
import math
import time
from collections import deque
from utils.counter import RepCounter
from utils.timer import DurationTimer  
from utils.angles import calculate_angle_3d, calculate_distance_3d

mp_pose = mp.solutions.pose.PoseLandmark

# ==================== AYARLAR (DÜZELTİLDİ) ====================
MOVEMENT_THRESHOLD = 0.01
LEAN_THRESHOLD = 15        # 20 → 15 (daha kolay)
HIP_STABILITY_THRESHOLD = 0.03  # 0.02 → 0.03 (daha toleranslı)
CIRCLE_DURATION = 5        # 15sn → 5sn (daha kısa)

# Açı eşikleri AZALTILDI
YANA_ACMA_THRESH = 60      # 85 → 60
ONE_ACMA_THRESH = 70       # 90 → 70
DISA_ACMA_THRESH = 40      # 50 → 40
DUVAR_YANA_THRESH = 60     # 80 → 60
DUVAR_ONE_THRESH = 65      # 85 → 65
DUVAR_GERIYE_THRESH = 30   # 40 → 30

MAX_REPS = 10

# ==================== GLOBAL STATE ====================
last_wrist_pos_sol = None
last_wrist_pos_sag = None
last_hip_x = None
last_hip_y = None

# Çember için
circle_start_time_sol = None
circle_start_time_sag = None
circle_completed_sol = False
circle_completed_sag = False

# ==================== SAYAÇLAR (Her taraf ayrı) ====================
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
    
    print("✅ Omuz modülü sıfırlandı.")

# ==================== YARDIMCI FONKSİYONLAR ====================
def check_visibility(landmarks, indices, threshold=0.3):
    """Görünürlük kontrolü - Eşik AZALTILDI (0.5 → 0.3)"""
    for idx in indices:
        if landmarks[idx].visibility < threshold:
            return False
    return True

def check_camera_angle(l_sh, r_sh, l_hip, r_hip):
    """
    Kamera açısı kontrolü
    Returns: "FRONT", "SIDE", "BACK"
    """
    # Omuz genişliği
    shoulder_width = abs(l_sh[0] - r_sh[0])
    
    # Yan duruş: Omuzlar üst üste (genişlik çok küçük)
    if shoulder_width < 0.15:
        return "SIDE"
    
    # Arka duruş: Omuzlar görünüyor ama elbow/wrist görünmüyor
    # (Bu main'de kontrol edilecek)
    
    # Normal (ön) duruş
    return "FRONT"

def check_stability(l_hip, r_hip):
    """Kalça stabilite kontrolü - Daha toleranslı"""
    global last_hip_x, last_hip_y
    current_x = (l_hip[0] + r_hip[0]) / 2
    current_y = (l_hip[1] + r_hip[1]) / 2
    
    if last_hip_x is None:
        last_hip_x = current_x
        last_hip_y = current_y
        return True
    
    diff_x = abs(current_x - last_hip_x)
    diff_y = abs(current_y - last_hip_y)
    last_hip_x = current_x
    last_hip_y = current_y
    
    # Eşik artırıldı (0.008 → 0.012)
    if diff_x > 0.012 or diff_y > 0.012:
        return False
    return True

def did_hip_lift(current_hip_y):
    """Kalça kalkışı kontrolü"""
    global last_hip_y
    if last_hip_y is None:
        last_hip_y = current_hip_y
        return False
    
    diff = last_hip_y - current_hip_y
    last_hip_y = last_hip_y * 0.95 + current_hip_y * 0.05
    
    return diff > HIP_STABILITY_THRESHOLD

def calculate_trunk_lean(shoulder, hip):
    """Gövde öne eğilme açısı"""
    vertical_pt = [hip[0], hip[1] - 0.5, hip[2]]
    return calculate_angle_3d(shoulder, hip, vertical_pt)

def check_stick_hold(l_wrist, r_wrist):
    """Sopa tutuş kontrolü - Daha toleranslı"""
    dist = calculate_distance_3d(l_wrist, r_wrist)
    # Eşik gevşetildi (0.1-0.8 → 0.05-1.0)
    if dist < 0.05 or dist > 1.0:
        return False
    return True

def get_total_reps(counter_sol, counter_sag):
    """İki tarafın minimum tekrarını al"""
    return min(counter_sol.rep_count, counter_sag.rep_count)

# ==================== ANA FONKSİYON ====================
def get_exercise_feedback(current_exercise, landmarks):
    global last_wrist_pos_sol, last_wrist_pos_sag
    global circle_start_time_sol, circle_start_time_sag, circle_completed_sol, circle_completed_sag

    feedback_talimat = "Egzersiz secilmedi"
    feedback_mesaj = ""
    
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

        # Omuz görünürlük kontrolü
        if not check_visibility(landmarks, [IDX_L_SH, IDX_R_SH]):
            return feedback_talimat, "⚠️ Omuzlar gorunmeli!"

        # Kamera açısı tespiti
        camera_angle = check_camera_angle(l_sh, r_sh, l_hip, r_hip)

        # ==================== SOPA EGZERSİZLERİ ====================
        if "ACMA" in current_exercise and not ("DUVAR" in current_exercise):
            # El görünürlüğü (düşük eşik)
            if not check_visibility(landmarks, [IDX_L_WR, IDX_R_WR], threshold=0.2):
                feedback_mesaj = "⚠️ Ellerini goster!"
            elif not check_stick_hold(l_wr, r_wr):
                feedback_talimat = "Sopayi tut!"
                feedback_mesaj = "Eller aralikli olmali"
            elif not check_stability(l_hip, r_hip):
                feedback_mesaj = "Sabit dur"
            else:
                # ==================== YANA AÇMA ====================
                if current_exercise == "OMUZ_YANA_ACMA":
                    feedback_talimat = "1. Yana Acma: Sopayi yana kaldır (Her taraf 10'ar)"
                    
                    angle_sol = calculate_angle_3d(l_hip, l_sh, l_el)
                    angle_sag = calculate_angle_3d(r_hip, r_sh, r_el)
                    
                    sol_reps = counter_yana_acma_sol.rep_count
                    sag_reps = counter_yana_acma_sag.rep_count
                    
                    if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                        feedback_mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                    else:
                        msg_sol = counter_yana_acma_sol.count(angle_sol)
                        msg_sag = counter_yana_acma_sag.count(angle_sag)
                        feedback_mesaj = f"Açı: {int(angle_sol)}°/{int(angle_sag)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"

                # ==================== ÖNE AÇMA ====================
                elif current_exercise == "OMUZ_ONE_ACMA":
                    feedback_talimat = "3. One Acma: Sopayi one kaldır (Her taraf 10'ar)"
                    
                    hip_center_y = (l_hip[1] + r_hip[1]) / 2
                    if did_hip_lift(hip_center_y):
                        feedback_mesaj = "⚠️ BEL KALKTI! Daha az yukari"
                    else:
                        angle_sol = calculate_angle_3d(l_hip, l_sh, l_el)
                        angle_sag = calculate_angle_3d(r_hip, r_sh, r_el)
                        
                        sol_reps = counter_one_acma_sol.rep_count
                        sag_reps = counter_one_acma_sag.rep_count
                        
                        if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                            feedback_mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                        else:
                            msg_sol = counter_one_acma_sol.count(angle_sol)
                            msg_sag = counter_one_acma_sag.count(angle_sag)
                            feedback_mesaj = f"Açı: {int(angle_sol)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"

                # ==================== DIŞA AÇMA ====================
                elif current_exercise == "OMUZ_DISA_ACMA":
                    feedback_talimat = "2. Disa Acma: Dirsek 90°, elleri disa ac (Her taraf 10'ar)"
                    
                    elbow_angle = calculate_angle_3d(l_sh, l_el, l_wr)
                    if elbow_angle < 50 or elbow_angle > 140:
                        feedback_mesaj = f"⚠️ Dirsek 90° olsun (Şu an: {int(elbow_angle)}°)"
                    else:
                        rot_sol = calculate_angle_3d(l_hip, l_sh, l_wr)
                        rot_sag = calculate_angle_3d(r_hip, r_sh, r_wr)
                        
                        sol_reps = counter_disa_acma_sol.rep_count
                        sag_reps = counter_disa_acma_sag.rep_count
                        
                        if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                            feedback_mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                        else:
                            msg_sol = counter_disa_acma_sol.count(rot_sol)
                            msg_sag = counter_disa_acma_sag.count(rot_sag)
                            feedback_mesaj = f"Sol:{sol_reps}/10 Sağ:{sag_reps}/10"

        # ==================== PENDUL EGZERSİZLERİ ====================
        elif "PEN" in current_exercise or "CEMBER" in current_exercise:
            if not check_visibility(landmarks, [IDX_L_WR, IDX_R_WR], threshold=0.2):
                feedback_mesaj = "⚠️ Kolunu goster!"
            else:
                # Öne eğilme kontrolü
                lean_angle_sol = calculate_trunk_lean(l_sh, l_hip)
                lean_angle_sag = calculate_trunk_lean(r_sh, r_hip)
                avg_lean = (lean_angle_sol + lean_angle_sag) / 2
                
                if avg_lean < LEAN_THRESHOLD:
                    feedback_talimat = "One egilin!"
                    feedback_mesaj = f"Eğilme: {int(avg_lean)}° (Hedef: >{LEAN_THRESHOLD}°)"
                else:
                    # ==================== ÖNDE SALLAMA ====================
                    if current_exercise == "OMUZ_PEN_FLEKSIYON":
                        feedback_talimat = "4. Onde Sallama: One egik, one-arkaya salla (Her taraf 10'ar)"
                        
                        arm_angle_sol = calculate_angle_3d(l_hip, l_sh, l_wr)
                        arm_angle_sag = calculate_angle_3d(r_hip, r_sh, r_wr)
                        
                        sol_reps = counter_pen_fleksiyon_sol.rep_count
                        sag_reps = counter_pen_fleksiyon_sag.rep_count
                        
                        if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                            feedback_mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                        else:
                            msg_sol = counter_pen_fleksiyon_sol.count(arm_angle_sol)
                            msg_sag = counter_pen_fleksiyon_sag.count(arm_angle_sag)
                            feedback_mesaj = f"Salınım: {int(arm_angle_sol)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"

                    # ==================== YANDA SALLAMA ====================
                    elif current_exercise == "OMUZ_PEN_ABDUKSIYON":
                        feedback_talimat = "5. Yanda Sallama: Saga-sola salla (Her taraf 10'ar)"
                        
                        sol_reps = counter_pen_abduksiyon_sol.rep_count
                        sag_reps = counter_pen_abduksiyon_sag.rep_count
                        
                        if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                            feedback_mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                        else:
                            msg_sol = counter_pen_abduksiyon_sol.count(calculate_angle_3d(r_sh, l_sh, l_wr))
                            msg_sag = counter_pen_abduksiyon_sag.count(calculate_angle_3d(l_sh, r_sh, r_wr))
                            feedback_mesaj = f"Sol:{sol_reps}/10 Sağ:{sag_reps}/10"

                    # ==================== ÇEMBER (BASİTLEŞTİRİLDİ) ====================
                    elif current_exercise == "OMUZ_CEMBER":
                        feedback_talimat = f"6. Cember Cizme: One egik, {CIRCLE_DURATION}sn cember ciz"
                        
                        # Basit hareket kontrolü
                        is_moving_sol = False
                        is_moving_sag = False
                        
                        if last_wrist_pos_sol:
                            dist_sol = calculate_distance_3d(last_wrist_pos_sol, l_wr)
                            if dist_sol > 0.005:  # Hareket var
                                is_moving_sol = True
                        last_wrist_pos_sol = l_wr
                        
                        if last_wrist_pos_sag:
                            dist_sag = calculate_distance_3d(last_wrist_pos_sag, r_wr)
                            if dist_sag > 0.005:
                                is_moving_sag = True
                        last_wrist_pos_sag = r_wr
                        
                        # SOL KOL
                        if is_moving_sol and not circle_completed_sol:
                            if circle_start_time_sol is None:
                                circle_start_time_sol = time.time()
                            elapsed = time.time() - circle_start_time_sol
                            if elapsed >= CIRCLE_DURATION:
                                circle_completed_sol = True
                                msg_sol = "✅ TAMAM!"
                            else:
                                msg_sol = f"Çiz... {int(CIRCLE_DURATION - elapsed)}sn"
                        elif circle_completed_sol:
                            msg_sol = "✅ TAMAM!"
                        else:
                            msg_sol = "Başla"
                            circle_start_time_sol = None
                        
                        # SAĞ KOL
                        if is_moving_sag and not circle_completed_sag:
                            if circle_start_time_sag is None:
                                circle_start_time_sag = time.time()
                            elapsed = time.time() - circle_start_time_sag
                            if elapsed >= CIRCLE_DURATION:
                                circle_completed_sag = True
                                msg_sag = "✅ TAMAM!"
                            else:
                                msg_sag = f"Çiz... {int(CIRCLE_DURATION - elapsed)}sn"
                        elif circle_completed_sag:
                            msg_sag = "✅ TAMAM!"
                        else:
                            msg_sag = "Başla"
                            circle_start_time_sag = None
                        
                        feedback_mesaj = f"Sol: {msg_sol} | Sağ: {msg_sag}"

        # ==================== DUVAR EGZERSİZLERİ ====================
        elif "DUVAR" in current_exercise or "GERME" in current_exercise:
            if not check_stability(l_hip, r_hip):
                feedback_mesaj = "Sabit dur"
            elif not check_visibility(landmarks, [IDX_L_EL, IDX_R_EL], threshold=0.2):
                feedback_mesaj = "⚠️ Kolunu goster!"
            else:
                # Yan duruş kontrolü (bazı hareketler için)
                if camera_angle == "SIDE" and "YANA" not in current_exercise:
                    feedback_mesaj = "⚠️ Kameraya ON durun!"
                else:
                    # ==================== DUVARA YANA ====================
                    if current_exercise == "OMUZ_DUVAR_YANA":
                        feedback_talimat = "7. Duvara Yana: YAN dur, kolunu yana ac (Her taraf 10'ar)"
                        
                        angle_sol = calculate_angle_3d(l_hip, l_sh, l_el)
                        angle_sag = calculate_angle_3d(r_hip, r_sh, r_el)
                        
                        sol_reps = counter_duvar_yana_sol.rep_count
                        sag_reps = counter_duvar_yana_sag.rep_count
                        
                        if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                            feedback_mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                        else:
                            msg_sol = counter_duvar_yana_sol.count(angle_sol)
                            msg_sag = counter_duvar_yana_sag.count(angle_sag)
                            feedback_mesaj = f"Açı: {int(angle_sol)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"
                    
                    # ==================== DUVARA ÖNE ====================
                    elif current_exercise == "OMUZ_DUVAR_ONE":
                        feedback_talimat = "8. Duvara One: Kolunu one it (Her taraf 10'ar)"
                        
                        angle_sol = calculate_angle_3d(l_hip, l_sh, l_el)
                        angle_sag = calculate_angle_3d(r_hip, r_sh, r_el)
                        
                        sol_reps = counter_duvar_one_sol.rep_count
                        sag_reps = counter_duvar_one_sag.rep_count
                        
                        if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                            feedback_mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                        else:
                            msg_sol = counter_duvar_one_sol.count(angle_sol)
                            msg_sag = counter_duvar_one_sag.count(angle_sag)
                            feedback_mesaj = f"Açı: {int(angle_sol)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"
                    
                    # ==================== DUVARA GERİYE ====================
                    elif current_exercise == "OMUZ_DUVAR_GERIYE":
                        feedback_talimat = "9. Duvara Geriye: Kolunu geriye it (Her taraf 10'ar)"
                        
                        angle_sol = calculate_angle_3d(l_hip, l_sh, l_wr)
                        angle_sag = calculate_angle_3d(r_hip, r_sh, r_wr)
                        
                        sol_reps = counter_duvar_geriye_sol.rep_count
                        sag_reps = counter_duvar_geriye_sag.rep_count
                        
                        if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                            feedback_mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                        else:
                            msg_sol = counter_duvar_geriye_sol.count(angle_sol)
                            msg_sag = counter_duvar_geriye_sag.count(angle_sag)
                            feedback_mesaj = f"Açı: {int(angle_sol)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"
                    
                    # ==================== GERME ====================
                    elif current_exercise == "OMUZ_GERME":
                        feedback_talimat = "10. Germe (15sn): Kolunu gogse cek ve tut"
                        
                        # Kol göğüse yakın mı?
                        active_sol = calculate_angle_3d(l_el, l_sh, r_sh) < 50
                        active_sag = calculate_angle_3d(r_el, r_sh, l_sh) < 50
                        
                        msg_sol = timer_germe_sol.update_feedback(active_sol)
                        msg_sag = timer_germe_sag.update_feedback(active_sag)
                        
                        feedback_mesaj = f"Sol: {msg_sol} | Sağ: {msg_sag}"

    except Exception as e:
        feedback_mesaj = f"❌ Hata: {str(e)}"
        print(f"OMUZ MODULU HATA: {e}")

    return feedback_talimat, feedback_mesaj