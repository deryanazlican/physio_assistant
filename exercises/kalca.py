import numpy as np
import mediapipe as mp
import time
from utils.counter import RepCounter
from utils.angles import calculate_angle_3d

mp_pose = mp.solutions.pose.PoseLandmark

# ==================== AYARLAR (DÜŞÜRÜLDÜ) ====================
DIZ_CEKME_THRESH = 100      # 110 → 100
DUZ_KALDIR_THRESH = 140     # 150 → 140
KOPRU_THRESH = 150          # 160 → 150
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

# ==================== RESET ====================
def reset_kalca_counters():
    global yan_start_time_sol, yan_start_time_sag, yan_completed_sol, yan_completed_sag
    global yuzustu_start_time_sol, yuzustu_start_time_sag, yuzustu_completed_sol, yuzustu_completed_sag
    
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
    
    print("✅ Kalça modülü sıfırlandı.")

# ==================== YARDIMCI FONKSİYONLAR ====================
def get_lm(landmarks, lm_name):
    lm = landmarks[lm_name.value]
    if lm.visibility < 0.25:  # 0.3 → 0.25
        return None
    return [lm.x, lm.y, lm.z]

def check_side_lying(l_sh, r_sh):
    """Yan yatış kontrolü"""
    y_diff = abs(l_sh[1] - r_sh[1])
    return y_diff > 0.10  # 0.10 → Daha kolay

def get_top_leg(l_hip, r_hip):
    """Hangi kalça üstte?"""
    return "SOL" if l_hip[1] < r_hip[1] else "SAG"

def check_prone(l_sh, l_hip):
    """Yüzüstü kontrolü"""
    return abs(l_sh[1] - l_hip[1]) < 0.15

# ==================== ANA FONKSİYON ====================
def get_exercise_feedback(current_exercise, landmarks):
    global yan_start_time_sol, yan_start_time_sag, yan_completed_sol, yan_completed_sag
    global yuzustu_start_time_sol, yuzustu_start_time_sag, yuzustu_completed_sol, yuzustu_completed_sag
    
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

        if not l_hip or not r_hip or not l_knee:
            return "⚠️ Gorunmuyorsun", "Kameraya gec", {}

        # ==================== 1. DİZİ GÖĞSE ÇEKME ====================
        if current_exercise == "KALCA_DIZ_CEKME":
            talimat = "Sirtustu yat, dizini gogsune cek (Her bacak 10'ar)"
            
            angle_sol = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_sag = calculate_angle_3d(r_sh, r_hip, r_knee)
            
            sol_reps = counter_diz_cekme_sol.rep_count
            sag_reps = counter_diz_cekme_sag.rep_count
            
            if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                msg_sol = counter_diz_cekme_sol.count(angle_sol)
                msg_sag = counter_diz_cekme_sag.count(angle_sag)
                mesaj = f"{int(angle_sol)}°/{int(angle_sag)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"
                ekstra_bilgi = {"angle": min(angle_sol, angle_sag), "reps": min(sol_reps, sag_reps), "max_reps": MAX_REPS}

        # ==================== 2. DÜZ BACAK KALDIRMA ====================
        elif current_exercise == "KALCA_DUZ_KALDIR":
            talimat = "Dizini BUKMEDEN bacagini kaldir (Her bacak 10'ar)"
            
            # Diz açısı kontrolü
            knee_angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            knee_angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)
            
            # Kalça açısı
            hip_angle_sol = calculate_angle_3d(l_sh, l_hip, l_knee)
            hip_angle_sag = calculate_angle_3d(r_sh, r_hip, r_knee)
            
            sol_reps = counter_duz_kaldir_sol.rep_count
            sag_reps = counter_duz_kaldir_sag.rep_count
            
            if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                msg_sol = ""
                msg_sag = ""
                
                # SOL
                if knee_angle_sol < 140:  # 150 → 140
                    msg_sol = "Dizi duzelt!"
                else:
                    msg_sol = counter_duz_kaldir_sol.count(hip_angle_sol)
                
                # SAĞ
                if knee_angle_sag < 140:
                    msg_sag = "Dizi duzelt!"
                else:
                    msg_sag = counter_duz_kaldir_sag.count(hip_angle_sag)
                
                mesaj = f"Sol:{sol_reps}/10 ({msg_sol}) | Sağ:{sag_reps}/10 ({msg_sag})"
                ekstra_bilgi = {"reps": min(sol_reps, sag_reps), "max_reps": MAX_REPS}

        # ==================== 3. KÖPRÜ KURMA ====================
        elif current_exercise == "KALCA_KOPRU":
            talimat = "Kalçani havaya kaldir ve sik (10 tekrar)"
            
            angle_sol = calculate_angle_3d(l_sh, l_hip, l_knee)
            angle_sag = calculate_angle_3d(r_sh, r_hip, r_knee)
            avg_angle = (angle_sol + angle_sag) / 2
            
            reps = counter_kopru.rep_count
            
            if reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! ({MAX_REPS} tekrar)"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                msg = counter_kopru.count(avg_angle)
                mesaj = f"{int(avg_angle)}° | {reps}/10 | {msg}"
                ekstra_bilgi = {"angle": avg_angle, "reps": reps, "max_reps": MAX_REPS}

        # ==================== 4. YAN YATARAK AÇMA ====================
        elif current_exercise == "KALCA_YAN_ACMA":
            talimat = f"Yan yat, ustteki bacagi kaldir ({YAN_HOLD_TIME}sn her bacak)"
            
            if not check_side_lying(l_sh, r_sh):
                mesaj = "⚠️ Yan yatmalısın!"
                return talimat, mesaj, {}
            
            top_leg = get_top_leg(l_hip, r_hip)
            foot_spread = abs(l_ankle[1] - r_ankle[1])
            is_active = foot_spread > 0.12  # 0.15 → 0.12
            
            if top_leg == "SOL":
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

        # ==================== 5. YÜZÜSTÜ KALDIRMA ====================
        elif current_exercise == "KALCA_YUZUSTU":
            talimat = f"Yuzustu yat, bacagini geriye kaldir ({YUZUSTU_HOLD_TIME}sn her bacak)"
            
            if not check_prone(l_sh, l_hip):
                mesaj = "⚠️ Yüzüstü yatmalısın!"
                return talimat, mesaj, {}
            
            # Topuk kalçadan yukarıda mı?
            active_sol = l_ankle[1] < (l_hip[1] - 0.03)  # 0.05 → 0.03
            active_sag = r_ankle[1] < (r_hip[1] - 0.03)
            
            # SOL
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
            
            # SAĞ
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
            
            mesaj = f"Sol: {msg_sol} | Sağ: {msg_sag}"

        # ==================== 6. YAN DİZ ÇEKME ====================
        elif current_exercise == "KALCA_YAN_DIZ_CEKME":
            talimat = "Yan yatarken dizini karnina cek (Her bacak 10'ar)"
            
            if not check_side_lying(l_sh, r_sh):
                mesaj = "⚠️ Yan yatmalısın!"
                return talimat, mesaj, {}
            
            top_leg = get_top_leg(l_hip, r_hip)
            
            if top_leg == "SOL":
                angle = calculate_angle_3d(l_sh, l_hip, l_knee)
                reps = counter_yan_diz_sol.rep_count
                
                if reps >= MAX_REPS:
                    mesaj = f"✅ SOL TAMAM! ({MAX_REPS} tekrar)"
                else:
                    msg = counter_yan_diz_sol.count(angle)
                    mesaj = f"SOL: {reps}/10 ({int(angle)}°)"
                
                ekstra_bilgi = {"angle": angle, "reps": reps, "max_reps": MAX_REPS}
            else:
                angle = calculate_angle_3d(r_sh, r_hip, r_knee)
                reps = counter_yan_diz_sag.rep_count
                
                if reps >= MAX_REPS:
                    mesaj = f"✅ SAĞ TAMAM! ({MAX_REPS} tekrar)"
                else:
                    msg = counter_yan_diz_sag.count(angle)
                    mesaj = f"SAĞ: {reps}/10 ({int(angle)}°)"
                
                ekstra_bilgi = {"angle": angle, "reps": reps, "max_reps": MAX_REPS}

        else:
            mesaj = "Bilinmeyen hareket"

    except Exception as e:
        mesaj = f"❌ Hata: {str(e)}"
        print(f"KALCA MODULU HATA: {e}")

    return talimat, mesaj, ekstra_bilgi