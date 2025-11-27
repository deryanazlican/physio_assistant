# exercises/diz.py
# YENİ DOSYA - Diz Egzersizleri Mantığı

import numpy as np
import mediapipe as mp
from utils.counter import RepCounter
from utils.timer import DurationTimer
from utils.angles import calculate_angle_3d

# --- Landmark Sabitleri ---
mp_pose = mp.solutions.pose
TARGET_REPS = 10 # Her egzersiz için hedef tekrar sayısı

# --- Sayaçların Tanımlanması ---

# 1. Diz Germe (Oturarak)
# Açı: HIP-KNEE-ANKLE. Hedef: > 175 (düz), Nötr: < 160 (bükülü)
counter_diz_germe_sol = RepCounter("DIZ_GERME", "Sol", threshold_angle=175, target_reps=TARGET_REPS, neutral_threshold=160)
counter_diz_germe_sag = RepCounter("DIZ_GERME", "Sag", threshold_angle=175, target_reps=TARGET_REPS, neutral_threshold=160)

# 2. Topuk Kaydırma (Yatarak)
# Açı: HIP-KNEE-ANKLE. Açı KÜÇÜLÜR. (180 - açı) hilesi kullanılır.
# Nötr (Düz): 170 -> 180-170=10. Hedef (Bükülü): 90 -> 180-90=90
counter_topuk_kaydir_sol = RepCounter("DIZ_TOPUK_KAYDIR", "Sol", threshold_angle=90, target_reps=TARGET_REPS, neutral_threshold=20)
counter_topuk_kaydir_sag = RepCounter("DIZ_TOPUK_KAYDIR", "Sag", threshold_angle=90, target_reps=TARGET_REPS, neutral_threshold=20)

# 3. Düz Bacak Kaldırma (Yatarak)
# Açı: SHOULDER-HIP-KNEE. Açı KÜÇÜLÜR. (180 - açı) hilesi kullanılır.
# Nötr (Yerde): 175 -> 180-175=5. Hedef (Havada): 135 -> 180-135=45
counter_bacak_kaldir_sol = RepCounter("DIZ_BACAK_KALDIR", "Sol", threshold_angle=45, target_reps=TARGET_REPS, neutral_threshold=15)
counter_bacak_kaldir_sag = RepCounter("DIZ_BACAK_KALDIR", "Sag", threshold_angle=45, target_reps=TARGET_REPS, neutral_threshold=15)

# 4. Oturarak Bacak Uzatma
# Açı: HIP-KNEE-ANKLE. Hedef: > 160 (düz), Nötr: < 110 (bükülü)
counter_otur_uzat_sol = RepCounter("DIZ_OTUR_UZAT", "Sol", threshold_angle=160, target_reps=TARGET_REPS, neutral_threshold=110)
counter_otur_uzat_sag = RepCounter("DIZ_OTUR_UZAT", "Sag", threshold_angle=160, target_reps=TARGET_REPS, neutral_threshold=110)

# 5. Duvar Squat (SÜRE BAZLI)
# Hedef açı: 80-110 derece arası. Hedef süre: 10 saniye
timer_duvar_squat_sol = DurationTimer("DIZ_DUVAR_SQUAT", "Sol", target_duration=10)
timer_duvar_squat_sag = DurationTimer("DIZ_DUVAR_SQUAT", "Sag", target_duration=10)

# 6. Otur Kalk
# Açı: HIP-KNEE-ANKLE. Hedef: > 160 (ayakta), Nötr: < 110 (oturuyor)
counter_otur_kalk_sol = RepCounter("DIZ_OTUR_KALK", "Sol", threshold_angle=160, target_reps=TARGET_REPS, neutral_threshold=110)
counter_otur_kalk_sag = RepCounter("DIZ_OTUR_KALK", "Sag", threshold_angle=160, target_reps=TARGET_REPS, neutral_threshold=110)


def reset_diz_counters():
    """Tüm diz sayaçlarını ve zamanlayıcıları sıfırlar."""
    counter_diz_germe_sol.reset(); counter_diz_germe_sag.reset()
    counter_topuk_kaydir_sol.reset(); counter_topuk_kaydir_sag.reset()
    counter_bacak_kaldir_sol.reset(); counter_bacak_kaldir_sag.reset()
    counter_otur_uzat_sol.reset(); counter_otur_uzat_sag.reset()
    timer_duvar_squat_sol.reset(); timer_duvar_squat_sag.reset()
    counter_otur_kalk_sol.reset(); counter_otur_kalk_sag.reset()
    print("Tüm Diz sayaçları ve zamanlayıcıları sıfırlandı.")


def get_exercise_feedback(current_exercise, landmarks):
    """
    Seçilen diz egzersizine göre açıları hesaplar, sayaçları günceller
    ve kullanıcıya talimat/mesaj döndürür.
    """
    
    feedback_talimat = "Egzersiz seçilmedi"
    feedback_mesaj = ""
    
    try:
        # Gerekli landmark koordinatlarını al
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]

        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]

        
        # --- EGZERSİZ YÖNLENDİRİCİ ---
        
        if current_exercise == "MENU_DIZ":
            feedback_talimat = "Lutfen bir diz egzersizi secin"

        elif current_exercise == "DIZ_GERME":
            feedback_talimat = "1. Diz Germe: Dizinizi gererek bacaginizi kaldirin."
            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)
            msg_sol = counter_diz_germe_sol.count(angle_sol)
            msg_sag = counter_diz_germe_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "DIZ_TOPUK_KAYDIR":
            feedback_talimat = "2. Topuk Kaydirma: Yuzustu yatin, topugu kalcaya cekin."
            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)
            msg_sol = counter_topuk_kaydir_sol.count(180 - angle_sol) # Açı küçüldüğü için
            msg_sag = counter_topuk_kaydir_sag.count(180 - angle_sag) # 180'den çıkar
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "DIZ_BACAK_KALDIR":
            feedback_talimat = "3. Duz Bacak Kaldirma: Sirtustu yatin, duz bacagi kaldirin."
            angle_sol = calculate_angle_3d(l_shoulder, l_hip, l_knee)
            angle_sag = calculate_angle_3d(r_shoulder, r_hip, r_knee)
            msg_sol = counter_bacak_kaldir_sol.count(180 - angle_sol) # Açı küçüldüğü için
            msg_sag = counter_bacak_kaldir_sag.count(180 - angle_sag) # 180'den çıkar
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "DIZ_OTUR_UZAT":
            feedback_talimat = "4. Oturarak Bacak Uzatma: Sandalyede bacaginizi duz uzatin."
            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)
            msg_sol = counter_otur_uzat_sol.count(angle_sol)
            msg_sag = counter_otur_uzat_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"
            
        elif current_exercise == "DIZ_DUVAR_SQUAT":
            feedback_talimat = "5. Duvar Squat (10sn): Duvara yaslanin ve 90 derece comelin."
            
            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)
            
            # Sol bacak için pozisyon kontrolü
            is_active_sol = False; pose_feedback_sol = ""
            if angle_sol > 120: pose_feedback_sol = "Biraz Comel"
            elif angle_sol < 80: pose_feedback_sol = "Biraz Kalk"
            else: is_active_sol = True; pose_feedback_sol = "POZISYONU KORU"
            
            # Sağ bacak için pozisyon kontrolü
            is_active_sag = False; pose_feedback_sag = ""
            if angle_sag > 120: pose_feedback_sag = "Biraz Comel"
            elif angle_sag < 80: pose_feedback_sag = "Biraz Kalk"
            else: is_active_sag = True; pose_feedback_sag = "POZISYONU KORU"

            # Zamanlayıcıları güncelle
            msg_sol = timer_duvar_squat_sol.update_feedback(is_active_sol)
            msg_sag = timer_duvar_squat_sag.update_feedback(is_active_sag)
            
            feedback_mesaj = f"[SOL: {pose_feedback_sol} - {msg_sol}]  [SAG: {pose_feedback_sag} - {msg_sag}]"

        elif current_exercise == "DIZ_OTUR_KALK":
            feedback_talimat = "6. Otur Kalk: Sandalyeden kalkin ve tekrar oturun."
            angle_sol = calculate_angle_3d(l_hip, l_knee, l_ankle)
            angle_sag = calculate_angle_3d(r_hip, r_knee, r_ankle)
            msg_sol = counter_otur_kalk_sol.count(angle_sol)
            msg_sag = counter_otur_kalk_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"
        
    except Exception as e:
        feedback_talimat = "Kullanici algilanmadi veya hesaplama hatasi."
        feedback_mesaj = f"Hata: {e}"
        # Hata durumunda da zamanlayıcıları duraklat
        timer_duvar_squat_sol.update_feedback(False)
        timer_duvar_squat_sag.update_feedback(False)
        
    return feedback_talimat, feedback_mesaj