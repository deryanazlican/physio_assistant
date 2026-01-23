import mediapipe as mp
import time
import numpy as np
from utils.angles import calculate_distance_3d
from utils.counter import RepCounter

mp_pose = mp.solutions.pose.PoseLandmark

# ==================== SABITLER ====================
LATERAL_THRESH = 15       # 30 → 15 (çok ağrıtıyordu)
ROTATION_THRESH = 15      # 25 → 15 (daha kolay)
FLEX_THRESH = 15          # 20 → 15 (daha kolay)
NEUTRAL_THRESH = 8        # 12 → 8 (daha hassas merkez)

MAX_REPS = 10
ISOMETRIC_WORK_TIME = 10
ISOMETRIC_REST_TIME = 5
ISOMETRIC_SETS = 3
ROTATION_HOLD_TIME = 3
COUNTDOWN_TIME = 3

# ==================== GLOBAL STATE ====================
# Rotasyon
rot_start_time = None
rotation_held_side = None
rotation_completed_sides = {"sag": False, "sol": False}

# İzometrik (3 set sistemi)
izo_state = 0           # 0:Bekle, 1:Hazırlık, 2:Çalış, 3:Dinlen, 4:Tamamlandı
izo_timer_start = None
izo_current_set = 0
izo_completed = False

# Çember (tamamen yeni)
circle_phase = 0  # 0:Başla, 1:Sağ, 2:Arka, 3:Sol, 4:Ön
circle_count = 0
circle_direction = None

# Lateral
lateral_must_return_center = False
lateral_last_side = None

# Fleksiyon
flex_must_return_center = False
flex_last_direction = None

# ==================== SAYAÇLAR ====================
lateral_counter_sag = RepCounter("Yana Egilme", "Sag", LATERAL_THRESH, MAX_REPS, NEUTRAL_THRESH)
lateral_counter_sol = RepCounter("Yana Egilme", "Sol", LATERAL_THRESH, MAX_REPS, NEUTRAL_THRESH)

rotasyon_counter_sag = RepCounter("Donme", "Sag", ROTATION_THRESH, MAX_REPS, NEUTRAL_THRESH)
rotasyon_counter_sol = RepCounter("Donme", "Sol", ROTATION_THRESH, MAX_REPS, NEUTRAL_THRESH)

fleksiyon_counter = RepCounter("ROM Fleksiyon", "On", FLEX_THRESH, MAX_REPS, NEUTRAL_THRESH)
ekstansiyon_counter = RepCounter("ROM Ekstansiyon", "Arka", FLEX_THRESH, MAX_REPS, NEUTRAL_THRESH)

circle_counter = RepCounter("Boyun Cember", "Tam Tur", 1, MAX_REPS, 0)

# ==================== YARDIMCI FONKSİYONLAR ====================
def get_lm(landmarks, lm_name):
    lm = landmarks[lm_name.value]
    if lm.visibility < 0.5:  # Görünürlük eşiği artırıldı
        return None
    return [lm.x, lm.y, lm.z]

def calculate_lateral_angle(nose, l_sh, r_sh):
    """Yana eğilme açısı"""
    neck_center = np.array([
        (l_sh[0] + r_sh[0]) / 2.0,
        (l_sh[1] + r_sh[1]) / 2.0
    ])

    head_vec = np.array([nose[0], nose[1]]) - neck_center
    vertical = np.array([0.0, -1.0])

    norm_head = np.linalg.norm(head_vec)
    if norm_head == 0:
        return 0.0

    dot = np.dot(head_vec, vertical)
    cosv = np.clip(dot / (norm_head * 1.0), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosv))

    if nose[0] > neck_center[0]:
        return angle
    else:
        return -angle

def calculate_flexion_angle(nose, l_sh, r_sh):
    """Öne/arkaya eğilme açısı - TAMAMEN YENİ HESAPLAMA"""
    shoulder_center_y = (l_sh[1] + r_sh[1]) / 2.0
    
    # Y farkı
    y_diff = nose[1] - shoulder_center_y
    
    # ÇOK DÜŞÜK ÇARPAN (10-15° başlasın)
    angle = y_diff * 80  # 150 → 80 
    
    return angle

def get_total_reps(exercise_name):
    """Toplam tekrar sayısı"""
    if exercise_name == "ROM_LAT":
        return min(lateral_counter_sag.rep_count, lateral_counter_sol.rep_count)
    elif exercise_name == "ROM_ROT":
        return min(rotasyon_counter_sag.rep_count, rotasyon_counter_sol.rep_count)
    elif exercise_name == "ROM_FLEKS":
        return min(fleksiyon_counter.rep_count, ekstansiyon_counter.rep_count)
    elif exercise_name == "ROM_CEMBER":
        return circle_count
    return 0

# ==================== RESET ====================
def reset_boyun_counters():
    global izo_state, izo_timer_start, izo_completed, izo_current_set
    global rot_start_time, rotation_held_side, rotation_completed_sides
    global circle_phase, circle_count, circle_direction
    global lateral_must_return_center, lateral_last_side
    global flex_must_return_center, flex_last_direction
    
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
    rotation_completed_sides = {"sag": False, "sol": False}
    
    circle_phase = 0
    circle_count = 0
    circle_direction = None
    
    lateral_must_return_center = False
    lateral_last_side = None
    
    flex_must_return_center = False
    flex_last_direction = None
    
    print("✅ Boyun modülü sıfırlandı.")

# ==================== İZOMETRİK (SADECE AÇI, EL YOK!) ====================
def process_isometric_3sets(head_angle):
    """
    3 SET SİSTEMİ - SADECE BAŞ AÇISI!
    head_angle > 10 ise çalışıyor kabul et
    """
    global izo_state, izo_timer_start, izo_completed, izo_current_set

    if izo_completed:
        return "✅ TAMAMLANDI! (3 SET)", {"completed": True, "progress": 100, "timer": 0, "set": 3}

    # BAŞ EĞİK DEĞİLSE DUR!
    if abs(head_angle) < 10:
        izo_state = 0
        izo_timer_start = None
        return "❌ Basini egik tut!", {"completed": False, "progress": 0, "timer": 0, "set": izo_current_set}

    # State 0: Başlangıç
    if izo_state == 0:
        izo_state = 1
        izo_timer_start = time.time()
        izo_current_set = 1
        return f"⏳ Hazirlan (3 sn)...", {"completed": False, "progress": 0, "timer": 3, "set": 1}
    
    # State 1: Hazırlık (3 sn)
    elif izo_state == 1:
        elapsed = time.time() - izo_timer_start
        remaining = COUNTDOWN_TIME - elapsed
        
        if elapsed < COUNTDOWN_TIME:
            progress = (elapsed / COUNTDOWN_TIME) * 100
            return f"⏳ Hazirlan... {int(remaining)}", {"completed": False, "progress": progress, "timer": remaining, "set": izo_current_set}
        else:
            izo_state = 2
            izo_timer_start = time.time()
            return "🔥 BASLA! TUT!", {"completed": False, "progress": 0, "timer": ISOMETRIC_WORK_TIME, "set": izo_current_set}
    
    # State 2: Çalışma (10 sn)
    elif izo_state == 2:
        elapsed = time.time() - izo_timer_start
        remaining = ISOMETRIC_WORK_TIME - elapsed
        progress = (elapsed / ISOMETRIC_WORK_TIME) * 100
        
        if elapsed < ISOMETRIC_WORK_TIME:
            return f"💪 TUT! {int(remaining)}sn | SET {izo_current_set}/3", {"completed": False, "progress": progress, "timer": remaining, "set": izo_current_set}
        else:
            if izo_current_set >= ISOMETRIC_SETS:
                izo_completed = True
                return "✅ HARIKA! 3 SET TAMAMLANDI!", {"completed": True, "progress": 100, "timer": 0, "set": 3}
            else:
                izo_state = 3
                izo_timer_start = time.time()
                return f"😌 DINLEN! {ISOMETRIC_REST_TIME}sn", {"completed": False, "progress": 0, "timer": ISOMETRIC_REST_TIME, "set": izo_current_set}
    
    # State 3: Dinlenme (5 sn)
    elif izo_state == 3:
        elapsed = time.time() - izo_timer_start
        remaining = ISOMETRIC_REST_TIME - elapsed
        progress = (elapsed / ISOMETRIC_REST_TIME) * 100
        
        if elapsed < ISOMETRIC_REST_TIME:
            return f"😌 Dinlen... {int(remaining)}sn | SET {izo_current_set}/3", {"completed": False, "progress": progress, "timer": remaining, "set": izo_current_set}
        else:
            izo_current_set += 1
            izo_state = 2
            izo_timer_start = time.time()
            return f"🔥 SET {izo_current_set} BASLA!", {"completed": False, "progress": 0, "timer": ISOMETRIC_WORK_TIME, "set": izo_current_set}
    
    return "", {"completed": False, "progress": 0, "timer": 0, "set": izo_current_set}

# ==================== ANA FONKSİYON ====================
def get_exercise_feedback(exercise_name, landmarks):
    global rot_start_time, rotation_held_side, rotation_completed_sides
    global circle_phase, circle_count, circle_direction
    global lateral_must_return_center, lateral_last_side
    global flex_must_return_center, flex_last_direction

    talimat = ""
    mesaj = ""
    ekstra_bilgi = {}

    try:
        nose = get_lm(landmarks, mp_pose.NOSE)
        l_sh = get_lm(landmarks, mp_pose.LEFT_SHOULDER)
        r_sh = get_lm(landmarks, mp_pose.RIGHT_SHOULDER)
        l_ear = get_lm(landmarks, mp_pose.LEFT_EAR)
        r_ear = get_lm(landmarks, mp_pose.RIGHT_EAR)
        l_wrist = get_lm(landmarks, mp_pose.LEFT_WRIST)
        r_wrist = get_lm(landmarks, mp_pose.RIGHT_WRIST)

        if not nose or not l_sh or not r_sh:
            return "⚠️ Gorunmuyorsun", "Kameraya tam karsidan gec", {}

        # ==================== ROM LATERAL (AYRI SAYIM) ====================
        if exercise_name == "ROM_LAT":
            talimat = "Her iki tarafa da 10'ar tekrar yap"
            tilt_angle = calculate_lateral_angle(nose, l_sh, r_sh)
            abs_angle = abs(tilt_angle)
            
            sag_reps = lateral_counter_sag.rep_count
            sol_reps = lateral_counter_sol.rep_count

            if sag_reps >= MAX_REPS and sol_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sağ:{sag_reps} Sol:{sol_reps})"
                ekstra_bilgi = {"angle": abs_angle, "reps": MAX_REPS, "max_reps": MAX_REPS, "completed": True}
                return talimat, mesaj, ekstra_bilgi

            if abs_angle < NEUTRAL_THRESH:
                lateral_counter_sag.count(0)
                lateral_counter_sol.count(0)
                mesaj = f"✓ MERKEZ | Sağ:{sag_reps}/10 Sol:{sol_reps}/10"
            elif tilt_angle > LATERAL_THRESH:
                msg = lateral_counter_sag.count(abs_angle)
                if "HARIKA" in msg:
                    mesaj = f"➡️ SAGA {int(abs_angle)}° | {msg}"
                elif lateral_counter_sag.state == "up":
                    mesaj = f"➡️ SAGA {int(abs_angle)}° | Merkeze don"
                else:
                    mesaj = f"➡️ SAGA {int(abs_angle)}° | Sağ:{sag_reps}/10"
            elif tilt_angle < -LATERAL_THRESH:
                msg = lateral_counter_sol.count(abs_angle)
                if "HARIKA" in msg:
                    mesaj = f"⬅️ SOLA {int(abs_angle)}° | {msg}"
                elif lateral_counter_sol.state == "up":
                    mesaj = f"⬅️ SOLA {int(abs_angle)}° | Merkeze don"
                else:
                    mesaj = f"⬅️ SOLA {int(abs_angle)}° | Sol:{sol_reps}/10"
            else:
                mesaj = f"Daha fazla egil | Sağ:{sag_reps}/10 Sol:{sol_reps}/10"

            ekstra_bilgi = {"angle": abs_angle, "reps": min(sag_reps, sol_reps), "max_reps": MAX_REPS}

        # ==================== ROM ROTASYON (ÇOK KOLAY) ====================
        elif exercise_name == "ROM_ROT":
            talimat = f"Yana dön ve {ROTATION_HOLD_TIME}sn tut"
            
            # Basit X koordinatı kontrolü
            center_x = (l_sh[0] + r_sh[0]) / 2.0
            head_x = nose[0]
            
            # Basit offset (omuz genişliği normalizasyonu yok!)
            offset = head_x - center_x
            
            sag_reps = rotasyon_counter_sag.rep_count
            sol_reps = rotasyon_counter_sol.rep_count

            if sag_reps >= MAX_REPS and sol_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Sağ:{sag_reps} Sol:{sol_reps})"
                ekstra_bilgi = {"completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
                return talimat, mesaj, ekstra_bilgi

            # ÇOK DÜŞÜK EŞİK (0.08 = çok kolay!)
            side = None
            if offset > 0.08:
                side = "sag"
            elif offset < -0.08:
                side = "sol"

            if side:
                if rot_start_time is None or side != rotation_held_side:
                    rot_start_time = time.time()
                    rotation_held_side = side

                elapsed = time.time() - rot_start_time
                remaining = ROTATION_HOLD_TIME - elapsed
                progress = (elapsed / ROTATION_HOLD_TIME) * 100

                if elapsed >= ROTATION_HOLD_TIME:
                    if not rotation_completed_sides[side]:
                        if side == "sag":
                            rotasyon_counter_sag.count(100)
                        else:
                            rotasyon_counter_sol.count(100)
                        rotation_completed_sides[side] = True
                        mesaj = f"✅ {side.upper()} TAMAM! | Sağ:{sag_reps}/10 Sol:{sol_reps}/10"
                    else:
                        mesaj = f"✓ {side.upper()} yapıldı | Sağ:{sag_reps}/10 Sol:{sol_reps}/10"
                else:
                    mesaj = f"🔄 {side.upper()} TUT! {int(remaining)}sn | Sağ:{sag_reps}/10 Sol:{sol_reps}/10"
                
                ekstra_bilgi = {"timer": remaining, "progress": progress, "reps": min(sag_reps, sol_reps), "max_reps": MAX_REPS}
            else:
                rot_start_time = None
                rotation_held_side = None
                if rotation_completed_sides["sag"] and rotation_completed_sides["sol"]:
                    rotation_completed_sides = {"sag": False, "sol": False}
                mesaj = f"↔️ MERKEZ | Sağ:{sag_reps}/10 Sol:{sol_reps}/10"
                ekstra_bilgi = {"reps": min(sag_reps, sol_reps), "max_reps": MAX_REPS}

        # ==================== ROM FLEKSİYON (DÜZELTİLDİ) ====================
        elif exercise_name == "ROM_FLEKS":
            talimat = "Ceneyi gogse gotur, sonra tavana bak (Her ikisi 10'ar)"
            
            # İyileştirilmiş açı hesabı
            angle = calculate_flexion_angle(nose, l_sh, r_sh)
            
            flex_reps = fleksiyon_counter.rep_count
            ext_reps = ekstansiyon_counter.rep_count

            if flex_reps >= MAX_REPS and ext_reps >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! (Öne:{flex_reps} Arkaya:{ext_reps})"
                ekstra_bilgi = {"angle": abs(angle), "completed": True, "reps": MAX_REPS, "max_reps": MAX_REPS}
                return talimat, mesaj, ekstra_bilgi

            if abs(angle) < NEUTRAL_THRESH:
                fleksiyon_counter.count(0)
                ekstansiyon_counter.count(0)
                mesaj = f"✓ MERKEZ | Öne:{flex_reps}/10 Arkaya:{ext_reps}/10"
            elif angle > FLEX_THRESH:
                msg = fleksiyon_counter.count(abs(angle))
                if "HARIKA" in msg:
                    mesaj = f"⬇️ FLEKSIYON {int(abs(angle))}° | {msg}"
                elif fleksiyon_counter.state == "up":
                    mesaj = f"⬇️ FLEKSIYON {int(abs(angle))}° | Merkeze don"
                else:
                    mesaj = f"⬇️ FLEKSIYON {int(abs(angle))}° | Öne:{flex_reps}/10"
            elif angle < -FLEX_THRESH:
                msg = ekstansiyon_counter.count(abs(angle))
                if "HARIKA" in msg:
                    mesaj = f"⬆️ EKSTANSIYON {int(abs(angle))}° | {msg}"
                elif ekstansiyon_counter.state == "up":
                    mesaj = f"⬆️ EKSTANSIYON {int(abs(angle))}° | Merkeze don"
                else:
                    mesaj = f"⬆️ EKSTANSIYON {int(abs(angle))}° | Arkaya:{ext_reps}/10"
            else:
                mesaj = f"Daha fazla egil | Öne:{flex_reps}/10 Arkaya:{ext_reps}/10"

            ekstra_bilgi = {"angle": abs(angle), "reps": min(flex_reps, ext_reps), "max_reps": MAX_REPS}

        # ==================== ÇEMBER (ÇOK BASİT!) ====================
        elif exercise_name == "ROM_CEMBER":
            talimat = "Basini saat yonunde cevir (10 tur)"
            
            # Basit koordinat kontrolü
            dx = nose[0] - ((l_sh[0] + r_sh[0]) / 2.0)
            dy = nose[1] - ((l_sh[1] + r_sh[1]) / 2.0)
            
            if circle_count >= MAX_REPS:
                mesaj = f"✅ TAMAMLANDI! ({MAX_REPS} tur)"
                ekstra_bilgi = {"completed": True, "reps": circle_count, "max_reps": MAX_REPS}
                return talimat, mesaj, ekstra_bilgi

            # ÇOK BASİT MANTIK
            if circle_phase == 0:  # ÖN
                if dy > 0.01:
                    circle_phase = 1
                    mesaj = f"✓ ÖN | {circle_count}/10"
                else:
                    mesaj = f"Öne egil | {circle_count}/10"
            
            elif circle_phase == 1:  # SAĞ
                if dx > 0.08:
                    circle_phase = 2
                    mesaj = f"✓ SAĞ | {circle_count}/10"
                else:
                    mesaj = f"Saga don | {circle_count}/10"
            
            elif circle_phase == 2:  # ARKA
                if dy < -0.01:
                    circle_phase = 3
                    mesaj = f"✓ ARKA | {circle_count}/10"
                else:
                    mesaj = f"Arkaya | {circle_count}/10"
            
            elif circle_phase == 3:  # SOL
                if dx < -0.08:
                    circle_phase = 4
                    mesaj = f"✓ SOL | {circle_count}/10"
                else:
                    mesaj = f"Sola don | {circle_count}/10"
            
            elif circle_phase == 4:  # ÖNE DÖNÜŞ
                if dy > 0.01:
                    circle_count += 1
                    circle_phase = 1
                    mesaj = f"✅ TUR {circle_count}! | {circle_count}/10"
                else:
                    mesaj = f"One gel | {circle_count}/10"

            ekstra_bilgi = {"reps": circle_count, "max_reps": MAX_REPS}

        # ==================== İZOMETRİK ÖN (SADECE AÇI!) ====================
        elif exercise_name == "IZO_FLEKS":
            talimat = "Elleri alnina koy ve one it (3x10sn)"
            
            # SADECE AÇI KONTROLÜ (el yok!)
            angle = calculate_flexion_angle(nose, l_sh, r_sh)
            
            # Öne eğiliyorsa (pozitif açı)
            mesaj, ekstra_bilgi = process_isometric_3sets(angle)

        # ==================== İZOMETRİK ARKA (SADECE AÇI!) ====================
        elif exercise_name == "IZO_EKST":
            talimat = "Elleri ensene koy ve geriye it (3x10sn)"
            
            # SADECE AÇI KONTROLÜ
            angle = calculate_flexion_angle(nose, l_sh, r_sh)
            
            # Geriye eğiliyorsa (negatif açı)
            mesaj, ekstra_bilgi = process_isometric_3sets(angle)

        # ==================== İZOMETRİK YAN (SADECE AÇI!) ====================
        elif exercise_name == "IZO_LAT":
            talimat = "Eli sakagina koy ve yana it (3x10sn)"
            
            # SADECE AÇI KONTROLÜ
            angle = calculate_lateral_angle(nose, l_sh, r_sh)
            
            # Yana eğiliyorsa
            mesaj, ekstra_bilgi = process_isometric_3sets(angle)

        else:
            talimat = "⚠️ Bilinmeyen Hareket"
            mesaj = f"Gelen: {exercise_name}"

    except Exception as e:
        mesaj = f"❌ Hata: {str(e)}"
        print(f"BOYUN MODULU HATA: {e}")

    return talimat, mesaj, ekstra_bilgi