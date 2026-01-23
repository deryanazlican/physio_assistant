import numpy as np
import mediapipe as mp
import time
from utils.counter import RepCounter
from utils.timer import DurationTimer
from utils.angles import calculate_angle_3d

mp_pose = mp.solutions.pose.PoseLandmark

# ==================== AYARLAR (DÜŞÜRÜLDÜ) ====================
HAVLU_EZME_ANGLE = 150      # 160 → 150
YUZUSTU_FLEX_THRESH = 110   # 100 → 110 (daha kolay)
YAN_KALDIR_LIMIT = 150      # 160 → 150
OTUR_UZAT_THRESH = 150      # 160 → 150
SQUAT_MIN = 70              # 85 → 70
SQUAT_MAX = 140             # 150 → 140

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

# ==================== RESET ====================
def reset_diz_counters():
    global last_hip_y
    global havlu_start_time_sol, havlu_start_time_sag, havlu_completed_sol, havlu_completed_sag
    global yan_start_time_sol, yan_start_time_sag, yan_completed_sol, yan_completed_sag
    global squat_start_time, squat_completed
    
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
    
    print("✅ Diz modülü sıfırlandı.")

# ==================== YARDIMCI FONKSİYONLAR ====================
def get_lm(landmarks, lm_name):
    lm = landmarks[lm_name.value]
    if lm.visibility < 0.25:  # 0.3 → 0.25
        return None
    return [lm.x, lm.y, lm.z]

def check_side_lying(l_sh, r_sh):
    """Yan yatış kontrolü"""
    y_diff = abs(l_sh[1] - r_sh[1])
    return y_diff > 0.12  # 0.15 → 0.12 (daha kolay)

def check_prone(l_sh, l_hip):
    """Yüzüstü kontrolü"""
    y_diff = abs(l_sh[1] - l_hip[1])
    return y_diff < 0.15  # Omuz ve kalça aynı hizada

# ==================== ANA FONKSİYON ====================
def get_exercise_feedback(current_exercise, landmarks):
    global last_hip_y
    global havlu_start_time_sol, havlu_start_time_sag, havlu_completed_sol, havlu_completed_sag
    global yan_start_time_sol, yan_start_time_sag, yan_completed_sol, yan_completed_sag
    global squat_start_time, squat_completed
    
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

        if not l_hip or not r_hip or not l_knee or not r_knee:
            return "⚠️ Gorunmuyorsun", "Bacaklarini goster", {}

        # ==================== 1. HAVLU EZME ====================
        if current_exercise == "DIZ_HAVLU_EZME":
            talimat = f"Dizin altina havlu koy, ez ve tut ({HAVLU_HOLD_TIME}sn her bacak)"
            
            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)
            
            # SOL BACAK
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
            
            # SAĞ BACAK
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
            
            mesaj = f"Sol: {msg_sol} | Sağ: {msg_sag}"
            ekstra_bilgi = {"angle": min(angle_sol, angle_sag)}

        # ==================== 2. YÜZÜSTÜ BÜKME ====================
        elif current_exercise == "DIZ_YUZUSTU_BUKME":
            talimat = "Yuzustu yat, topugunu kalcana cek (Her bacak 10'ar)"
            
            if not check_prone(l_sh, l_hip):
                mesaj = "⚠️ Yüzüstü yatmalısın!"
                return talimat, mesaj, {}
            
            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)
            
            sol_reps = counter_yuzustu_sol.rep_count
            sag_reps = counter_yuzustu_sag.rep_count
            
            if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                msg_sol = counter_yuzustu_sol.count(angle_sol)
                msg_sag = counter_yuzustu_sag.count(angle_sag)
                
                if "HARIKA" in msg_sol or "HARIKA" in msg_sag:
                    mesaj = f"Sol:{sol_reps}/10 Sağ:{sag_reps}/10"
                else:
                    mesaj = f"{int(angle_sol)}°/{int(angle_sag)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"
                
                ekstra_bilgi = {"angle": min(angle_sol, angle_sag), "reps": min(sol_reps, sag_reps), "max_reps": MAX_REPS}

        # ==================== 3. YAN YATARAK KALDIRMA ====================
        elif current_exercise == "DIZ_YAN_KALDIR":
            talimat = f"Yan yat, ustteki bacagi kaldir ({YAN_HOLD_TIME}sn her bacak)"
            
            if not check_side_lying(l_sh, r_sh):
                mesaj = "⚠️ Yan yatmalısın!"
                return talimat, mesaj, {}
            
            # Hangi taraf üstte?
            left_is_up = l_hip[1] < r_hip[1]
            
            if left_is_up:
                leg_spread = abs(l_ankle[1] - r_ankle[1])
                is_holding = leg_spread > 0.12  # 0.15 → 0.12
                
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

        # ==================== 4. OTURARAK UZATMA ====================
        elif current_exercise == "DIZ_OTUR_UZAT":
            talimat = "Oturarak dizini duzelestir (Her bacak 10'ar)"
            
            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)
            
            sol_reps = counter_otur_sol.rep_count
            sag_reps = counter_otur_sag.rep_count
            
            if sol_reps >= MAX_REPS and sag_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sol:{sol_reps} Sağ:{sag_reps})"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
            else:
                msg_sol = counter_otur_sol.count(angle_sol)
                msg_sag = counter_otur_sag.count(angle_sag)
                mesaj = f"{int(angle_sol)}°/{int(angle_sag)}° | Sol:{sol_reps}/10 Sağ:{sag_reps}/10"
                ekstra_bilgi = {"angle": min(angle_sol, angle_sag), "reps": min(sol_reps, sag_reps), "max_reps": MAX_REPS}

        # ==================== 5. DUVAR SQUAT ====================
        elif current_exercise == "DIZ_DUVAR_SQUAT":
            talimat = f"Sirini duvara yasla, çömel ve tut ({SQUAT_HOLD_TIME}sn)"
            
            angle = calculate_angle_3d(l_hip, l_knee, l_ankle)
            
            in_position = SQUAT_MIN < angle < SQUAT_MAX
            
            if in_position and not squat_completed:
                if squat_start_time is None:
                    squat_start_time = time.time()
                elapsed = time.time() - squat_start_time
                if elapsed >= SQUAT_HOLD_TIME:
                    squat_completed = True
                    mesaj = "✅ HARIKA TAMAMLANDI!"
                    ekstra_bilgi = {"completed": True, "progress": 100}
                else:
                    progress = (elapsed / SQUAT_HOLD_TIME) * 100
                    mesaj = f"💪 TUT! {int(SQUAT_HOLD_TIME - elapsed)}sn | Açı: {int(angle)}°"
                    ekstra_bilgi = {"timer": SQUAT_HOLD_TIME - elapsed, "progress": progress, "angle": angle}
            elif squat_completed:
                mesaj = "✅ TAMAMLANDI!"
                ekstra_bilgi = {"completed": True, "progress": 100}
            else:
                squat_start_time = None
                if angle >= SQUAT_MAX:
                    mesaj = f"Daha fazla çömel ({int(angle)}°)"
                else:
                    mesaj = f"Çok indin! Biraz kalk ({int(angle)}°)"
                ekstra_bilgi = {"angle": angle}

        else:
            mesaj = "Bilinmeyen hareket"

    except Exception as e:
        mesaj = f"❌ Hata: {str(e)}"
        print(f"DIZ MODULU HATA: {e}")

    return talimat, mesaj, ekstra_bilgi