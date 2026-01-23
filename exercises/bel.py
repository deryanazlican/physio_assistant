import mediapipe as mp
import time
import numpy as np
from utils.counter import RepCounter
from utils.angles import calculate_angle_3d

mp_pose = mp.solutions.pose.PoseLandmark

# ==================== AYARLAR (DÜŞÜRÜLDÜ) ====================
TEK_DIZ_THRESH = 70        # 80 → 70
CIFT_DIZ_THRESH = 70       # 80 → 70
MEKIK_THRESH = 120         # 130 → 120
SLR_THRESH = 35            # 45 → 35
KOPRU_THRESH = 150         # Köprü açısı
KEDI_THRESH = 15           # 20 → 15
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

# ==================== RESET ====================
def reset_bel_counters():
    global kopru_start_time, kopru_completed, yuzustu_start_time, yuzustu_completed
    
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
    
    print("✅ Bel modülü sıfırlandı.")

# ==================== YARDIMCI FONKSİYONLAR ====================
def get_lm(landmarks, lm_name):
    lm = landmarks[lm_name.value]
    if lm.visibility < 0.25:  # 0.4/0.5 → 0.25
        return None
    return [lm.x, lm.y, lm.z]

def avg(a, b):
    return [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2]

# ==================== ANA FONKSİYON ====================
def get_exercise_feedback(current_exercise, landmarks):
    global kopru_start_time, kopru_completed, yuzustu_start_time, yuzustu_completed
    
    talimat = ""
    mesaj = ""
    ekstra_bilgi = {}

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
            return "⚠️ Gorunmuyorsun", "Kameraya gec", {}

        # ==================== 1. TEK DİZ ÇEKME ====================
        if current_exercise == "BEL_TEK_DIZ":
            talimat = "Tek dizi gogsune cek (Her bacak 10'ar)"
            
            angle_l = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_r = calculate_angle_3d(r_sh, r_hip, r_knee)
            
            # Hangisi bükülmüş? (180 - açı = fleksiyon)
            flex_l = 180 - angle_l
            flex_r = 180 - angle_r
            
            sol_reps = counter_tek_diz_sol.rep_count
            sag_reps = counter_tek_diz_sag.rep_count
            
            if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                # Tek bacak mı bükülmüş?
                if flex_l > 30 and flex_r < 20:  # Sol bükük, sağ düz
                    msg = counter_tek_diz_sol.count(flex_l)
                    mesaj = f"SOL: {sol_reps}/10 ({int(flex_l)}°)"
                elif flex_r > 30 and flex_l < 20:  # Sağ bükük, sol düz
                    msg = counter_tek_diz_sag.count(flex_r)
                    mesaj = f"SAĞ: {sag_reps}/10 ({int(flex_r)}°)"
                else:
                    counter_tek_diz_sol.count(0)
                    counter_tek_diz_sag.count(0)
                    mesaj = "Tek bacak bukulmeli!"
                
                ekstra_bilgi = {"reps": min(sol_reps, sag_reps), "max_reps": MAX_REPS}

        # ==================== 2. ÇİFT DİZ ÇEKME ====================
        elif current_exercise == "BEL_CIFT_DIZ":
            talimat = "Iki dizi birden gogsune cek (10 tekrar)"
            
            angle_l = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_r = calculate_angle_3d(r_sh, r_hip, r_knee)
            avg_angle = (angle_l + angle_r) / 2
            flex = 180 - avg_angle
            
            reps = counter_cift_diz.rep_count
            
            if reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! ({MAX_REPS} tekrar)"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                msg = counter_cift_diz.count(flex)
                mesaj = f"{reps}/10 | {int(flex)}°"
                ekstra_bilgi = {"angle": flex, "reps": reps, "max_reps": MAX_REPS}

        # ==================== 3. YARIM MEKİK ====================
        elif current_exercise == "BEL_MEKIK":
            talimat = "Omuzlarini kaldir, dizlere uzan (10 tekrar)"
            
            shoulder_mid = avg(l_sh, r_sh)
            hip_mid = avg(l_hip, r_hip)
            
            # Gövde açısı
            torso_vec = np.array([shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]])
            vert = np.array([0.0, -1.0])
            norm = np.linalg.norm(torso_vec)
            
            if norm == 0:
                torso_angle = 0
            else:
                dot = np.dot(torso_vec, vert) / norm
                torso_angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
            
            flex_amount = 90 - torso_angle
            
            reps = counter_mekik.rep_count
            
            if reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! ({MAX_REPS} tekrar)"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                if flex_amount > 10:  # 15 → 10
                    msg = counter_mekik.count(flex_amount)
                    mesaj = f"{reps}/10 | Fleksiyon: {int(flex_amount)}°"
                else:
                    counter_mekik.count(0)
                    mesaj = f"Daha fazla kaldir ({int(flex_amount)}°)"
                
                ekstra_bilgi = {"angle": flex_amount, "reps": reps, "max_reps": MAX_REPS}

        # ==================== 4. DÜZ BACAK KALDIRMA (SLR) ====================
        elif current_exercise == "BEL_SLR":
            talimat = "Diz bukmeden bacagi kaldir (Her bacak 10'ar)"
            
            # Diz açıları
            knee_l = calculate_angle_3d(l_hip, l_knee, l_ankle)
            knee_r = calculate_angle_3d(r_hip, r_knee, r_ankle)
            
            # Kalça açıları
            hip_angle_l = calculate_angle_3d(l_sh, l_hip, l_knee)
            hip_angle_r = calculate_angle_3d(r_sh, r_hip, r_knee)
            
            flex_l = 180 - hip_angle_l
            flex_r = 180 - hip_angle_r
            
            sol_reps = counter_slr_sol.rep_count
            sag_reps = counter_slr_sag.rep_count
            
            if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                # Sol bacak kontrolü
                if knee_r > 150 and flex_l > 25 and not (flex_r > 25):  # Sağ yerde, sol havada
                    if knee_l < 140:  # 160 → 140
                        msg_sol = "SOL DİZİ DÜZELT!"
                    else:
                        msg_sol = counter_slr_sol.count(flex_l)
                    mesaj = f"SOL: {sol_reps}/10 | {msg_sol}"
                
                # Sağ bacak kontrolü
                elif knee_l > 150 and flex_r > 25 and not (flex_l > 25):  # Sol yerde, sağ havada
                    if knee_r < 140:
                        msg_sag = "SAĞ DİZİ DÜZELT!"
                    else:
                        msg_sag = counter_slr_sag.count(flex_r)
                    mesaj = f"SAĞ: {sag_reps}/10 | {msg_sag}"
                
                else:
                    counter_slr_sol.count(0)
                    counter_slr_sag.count(0)
                    mesaj = "Tek bacak kaldirmali!"
                
                ekstra_bilgi = {"reps": min(sol_reps, sag_reps), "max_reps": MAX_REPS}

        # ==================== 5. KÖPRÜ ====================
        elif current_exercise == "BEL_KOPRU":
            talimat = f"Kalcani kaldir ve tut ({KOPRU_HOLD_TIME}sn)"
            
            angle_l = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_r = calculate_angle_3d(r_sh, r_hip, r_knee)
            avg_angle = (angle_l + angle_r) / 2
            
            is_up = avg_angle > KOPRU_THRESH
            
            if is_up and not kopru_completed:
                if kopru_start_time is None:
                    kopru_start_time = time.time()
                elapsed = time.time() - kopru_start_time
                if elapsed >= KOPRU_HOLD_TIME:
                    kopru_completed = True
                    mesaj = "✅ HARIKA TAMAMLANDI!"
                    ekstra_bilgi = {"completed": True, "progress": 100}
                else:
                    progress = (elapsed / KOPRU_HOLD_TIME) * 100
                    mesaj = f"💪 TUT! {int(KOPRU_HOLD_TIME - elapsed)}sn | {int(avg_angle)}°"
                    ekstra_bilgi = {"timer": KOPRU_HOLD_TIME - elapsed, "progress": progress, "angle": avg_angle}
            elif kopru_completed:
                mesaj = "✅ TAMAMLANDI!"
                ekstra_bilgi = {"completed": True, "progress": 100}
            else:
                kopru_start_time = None
                mesaj = f"Daha fazla kaldir ({int(avg_angle)}°)"
                ekstra_bilgi = {"angle": avg_angle}

        # ==================== 6. KEDİ - DEVE ====================
        elif current_exercise == "BEL_KEDI_DEVE":
            talimat = "Sirtini kambur yap - duz yap (10 tekrar)"
            
            shoulder_mid = avg(l_sh, r_sh)
            hip_mid = avg(l_hip, r_hip)
            
            # Omurga hareketi (Y farkı)
            spine_diff = (shoulder_mid[1] - hip_mid[1]) * 80  # 100 → 80
            
            reps = counter_kedi.rep_count
            
            if reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! ({MAX_REPS} tekrar)"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                msg = counter_kedi.count(abs(spine_diff))
                mesaj = f"{reps}/10 | Hareket: {int(abs(spine_diff))}"
                ekstra_bilgi = {"reps": reps, "max_reps": MAX_REPS}

        # ==================== 7. YÜZÜSTÜ DOĞRULMA ====================
        elif current_exercise == "BEL_YUZUSTU":
            talimat = f"Yuzustu yat, govdeni kaldir ({YUZUSTU_HOLD_TIME}sn)"
            
            # Omuz kalçadan yukarıda mı?
            is_lifted = (l_sh[1] < (l_hip[1] - 0.06)) and (r_sh[1] < (r_hip[1] - 0.06))  # 0.08 → 0.06
            
            if is_lifted and not yuzustu_completed:
                if yuzustu_start_time is None:
                    yuzustu_start_time = time.time()
                elapsed = time.time() - yuzustu_start_time
                if elapsed >= YUZUSTU_HOLD_TIME:
                    yuzustu_completed = True
                    mesaj = "✅ HARIKA TAMAMLANDI!"
                    ekstra_bilgi = {"completed": True, "progress": 100}
                else:
                    progress = (elapsed / YUZUSTU_HOLD_TIME) * 100
                    mesaj = f"💪 TUT! {int(YUZUSTU_HOLD_TIME - elapsed)}sn"
                    ekstra_bilgi = {"timer": YUZUSTU_HOLD_TIME - elapsed, "progress": progress}
            elif yuzustu_completed:
                mesaj = "✅ TAMAMLANDI!"
                ekstra_bilgi = {"completed": True, "progress": 100}
            else:
                yuzustu_start_time = None
                mesaj = "Daha fazla kaldir"

        else:
            mesaj = "Bilinmeyen hareket"

    except Exception as e:
        mesaj = f"❌ Hata: {str(e)}"
        print(f"BEL MODULU HATA: {e}")

    return talimat, mesaj, ekstra_bilgi