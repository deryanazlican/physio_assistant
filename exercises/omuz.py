# exercises/omuz.py
# GÜNCELLENMİŞ TAM KOD (No 14 Germe Dahil - EKSİKSİZ)

import numpy as np
import mediapipe as mp
from utils.counter import RepCounter
from utils.timer import DurationTimer  
from utils.angles import calculate_angle_3d, calculate_distance_3d

# --- Landmark Sabitleri ---
mp_pose = mp.solutions.pose

# --- HAREKET ALGILAMA ---
# Çember hareketi için son pozisyonları sakla
last_wrist_pos_sol = None
last_wrist_pos_sag = None
MOVEMENT_THRESHOLD = 0.02 # Bileğin hareket ettiğini anlamak için gereken min. mesafe (normalize)

# --- 1. Tekrar Sayaçları (Mevcut) ---
counter_yana_acma_sol = RepCounter("OMUZ_YANA_ACMA", "Sol", threshold_angle=90, target_reps=10, neutral_threshold=30)
counter_yana_acma_sag = RepCounter("OMUZ_YANA_ACMA", "Sag", threshold_angle=90, target_reps=10, neutral_threshold=30)
counter_one_acma_sol = RepCounter("OMUZ_ONE_ACMA", "Sol", threshold_angle=90, target_reps=10, neutral_threshold=30)
counter_one_acma_sag = RepCounter("OMUZ_ONE_ACMA", "Sag", threshold_angle=90, target_reps=10, neutral_threshold=30)
counter_disa_acma_sol = RepCounter("OMUZ_DISA_ACMA", "Sol", threshold_angle=60, target_reps=10, neutral_threshold=25)
counter_disa_acma_sag = RepCounter("OMUZ_DISA_ACMA", "Sag", threshold_angle=60, target_reps=10, neutral_threshold=25)
counter_arkaya_acma_sol = RepCounter("OMUZ_ARKAYA_ACMA", "Sol", threshold_angle=60, target_reps=10, neutral_threshold=30)
counter_arkaya_acma_sag = RepCounter("OMUZ_ARKAYA_ACMA", "Sag", threshold_angle=60, target_reps=10, neutral_threshold=30)
counter_ice_acma_sol = RepCounter("OMUZ_ICE_ACMA", "Sol", threshold_angle=170, target_reps=10, neutral_threshold=155)
counter_ice_acma_sag = RepCounter("OMUZ_ICE_ACMA", "Sag", threshold_angle=170, target_reps=10, neutral_threshold=155)
counter_pen_fleksiyon_sol = RepCounter("OMUZ_PEN_FLEKSIYON", "Sol", threshold_angle=70, target_reps=10, neutral_threshold=40)
counter_pen_fleksiyon_sag = RepCounter("OMUZ_PEN_FLEKSIYON", "Sag", threshold_angle=70, target_reps=10, neutral_threshold=40)
counter_pen_abduksiyon_sol = RepCounter("OMUZ_PEN_ABDUKSIYON", "Sol", threshold_angle=45, target_reps=10, neutral_threshold=20)
counter_pen_abduksiyon_sag = RepCounter("OMUZ_PEN_ABDUKSIYON", "Sag", threshold_angle=45, target_reps=10, neutral_threshold=20)
counter_duvar_yana_sol = RepCounter("OMUZ_DUVAR_YANA", "Sol", threshold_angle=90, target_reps=10, neutral_threshold=30)
counter_duvar_yana_sag = RepCounter("OMUZ_DUVAR_YANA", "Sag", threshold_angle=90, target_reps=10, neutral_threshold=30)
counter_duvar_one_sol = RepCounter("OMUZ_DUVAR_ONE", "Sol", threshold_angle=90, target_reps=10, neutral_threshold=30)
counter_duvar_one_sag = RepCounter("OMUZ_DUVAR_ONE", "Sag", threshold_angle=90, target_reps=10, neutral_threshold=30)
counter_duvar_geriye_sol = RepCounter("OMUZ_DUVAR_GERIYE", "Sol", threshold_angle=60, target_reps=10, neutral_threshold=30)
counter_duvar_geriye_sag = RepCounter("OMUZ_DUVAR_GERIYE", "Sag", threshold_angle=60, target_reps=10, neutral_threshold=30)
counter_duvar_disa_sol = RepCounter("OMUZ_DUVAR_DISA", "Sol", threshold_angle=60, target_reps=10, neutral_threshold=25)
counter_duvar_disa_sag = RepCounter("OMUZ_DUVAR_DISA", "Sag", threshold_angle=60, target_reps=10, neutral_threshold=25)

# --- 2. Süre Zamanlayıcıları ---
timer_cember_sol = DurationTimer("OMUZ_CEMBER", "Sol", target_duration=15)
timer_cember_sag = DurationTimer("OMUZ_CEMBER", "Sag", target_duration=15)

# YENİ ZAMANLAYICI (No. 14 Germe)
timer_germe_sol = DurationTimer("OMUZ_GERME", "Sol", target_duration=15)
timer_germe_sag = DurationTimer("OMUZ_GERME", "Sag", target_duration=15)


def reset_omuz_counters():
    """Tüm omuz sayaçlarını ve zamanlayıcıları sıfırlar."""
    global last_wrist_pos_sol, last_wrist_pos_sag
    
    # Sopa
    counter_yana_acma_sol.reset(); counter_yana_acma_sag.reset()
    counter_one_acma_sol.reset(); counter_one_acma_sag.reset()
    counter_disa_acma_sol.reset(); counter_disa_acma_sag.reset()
    counter_arkaya_acma_sol.reset(); counter_arkaya_acma_sag.reset()
    counter_ice_acma_sol.reset(); counter_ice_acma_sag.reset()
    # Pendül
    counter_pen_fleksiyon_sol.reset(); counter_pen_fleksiyon_sag.reset()
    counter_pen_abduksiyon_sol.reset(); counter_pen_abduksiyon_sag.reset()
    # Duvar
    counter_duvar_yana_sol.reset(); counter_duvar_yana_sag.reset()
    counter_duvar_one_sol.reset(); counter_duvar_one_sag.reset()
    counter_duvar_geriye_sol.reset(); counter_duvar_geriye_sag.reset()
    counter_duvar_disa_sol.reset(); counter_duvar_disa_sag.reset()
    
    # Zamanlayıcıları sıfırla
    timer_cember_sol.reset()
    timer_cember_sag.reset()
    timer_germe_sol.reset() # YENİ
    timer_germe_sag.reset() # YENİ
    
    # Pozisyon hafızasını sıfırla
    last_wrist_pos_sol = None
    last_wrist_pos_sag = None
    
    print("Tum Omuz sayaçları ve zamanlayıcıları sıfırlandı.")


def get_exercise_feedback(current_exercise, landmarks):
    """
    Seçilen omuz egzersizine göre açıları/hareketi hesaplar, sayaçları/zamanlayıcıları günceller
    ve kullanıcıya talimat/mesaj döndürür.
    """
    global last_wrist_pos_sol, last_wrist_pos_sag
    
    feedback_talimat = "Egzersiz seçilmedi"
    feedback_mesaj = ""
    
    try:
        # Gerekli landmark koordinatlarını al
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
        
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]

        
        # --- EGZERSİZ YÖNLENDİRİCİ ---
        
        # MENÜLER
        if current_exercise == "MENU_OMUZ":
            feedback_talimat = "Lutfen bir omuz egzersiz tipi secin"
        elif current_exercise == "MENU_OMUZ_SOPA":
            feedback_talimat = "Lutfen bir sopa egzersizi secin"
        elif current_exercise == "MENU_OMUZ_PEN":
            feedback_talimat = "Lutfen bir sallanma egzersizi secin"
        elif current_exercise == "MENU_OMUZ_DUVAR":
            feedback_talimat = "Lutfen bir duvar/germe egzersizi secin" # GÜNCELLENDİ

        # --- 1. SOPA EGZERSİZLERİ ---
        elif current_exercise == "OMUZ_YANA_ACMA":
            feedback_talimat = "2. Yana Acma: Sopayi iki yandan yukari kaldirin."
            angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_elbow)
            angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_elbow)
            msg_sol = counter_yana_acma_sol.count(angle_sol)
            msg_sag = counter_yana_acma_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "OMUZ_ONE_ACMA":
            feedback_talimat = "4. One Acma: Sopayi duz kollarla onden yukari kaldirin."
            angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_elbow)
            angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_elbow)
            msg_sol = counter_one_acma_sol.count(angle_sol)
            msg_sag = counter_one_acma_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "OMUZ_DISA_ACMA":
            feedback_talimat = "3. Disa Acma: Dirsekler yanda, 90 derece bukulu. Elleri disa acin."
            elbow_angle_sol = calculate_angle_3d(l_shoulder, l_elbow, l_wrist)
            elbow_angle_sag = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)
            if (elbow_angle_sol < 70 or elbow_angle_sol > 110) or (elbow_angle_sag < 70 or elbow_angle_sag > 110):
                feedback_mesaj = "POZISYON HATALI: Dirsekleri 90 derece bukun!"
            else:
                angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_wrist)
                angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_wrist)
                msg_sol = counter_disa_acma_sol.count(angle_sol)
                msg_sag = counter_disa_acma_sag.count(angle_sag)
                feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"
        
        elif current_exercise == "OMUZ_ARKAYA_ACMA":
            feedback_talimat = "5. Arkaya Acma: Sopayi arkada, duz kollarla geriye kaldirin."
            angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_wrist)
            angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_wrist)
            msg_sol = counter_arkaya_acma_sol.count(angle_sol)
            msg_sag = counter_arkaya_acma_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "OMUZ_ICE_ACMA":
            feedback_talimat = "6. Ice Acma: Dirsekler yanda, 90 derece bukulu. Elleri ice cevirin."
            elbow_angle_sol = calculate_angle_3d(l_shoulder, l_elbow, l_wrist)
            elbow_angle_sag = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)
            if (elbow_angle_sol < 70 or elbow_angle_sol > 110) or (elbow_angle_sag < 70 or elbow_angle_sag > 110):
                feedback_mesaj = "POZISYON HATALI: Dirsekleri 90 derece bukun!"
            else:
                angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_wrist)
                angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_wrist)
                msg_sol = counter_ice_acma_sol.count(180 - angle_sol)
                msg_sag = counter_ice_acma_sag.count(180 - angle_sag)
                feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"
        
        # --- 2. SALLANMA (PENDÜL) EGZERSİZLERİ ---
        elif current_exercise == "OMUZ_PEN_FLEKSIYON":
            feedback_talimat = "7. Onde Sallama: One egilin, kolu onden arkaya sallayin."
            angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_wrist)
            angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_wrist)
            msg_sol = counter_pen_fleksiyon_sol.count(angle_sol)
            msg_sag = counter_pen_fleksiyon_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "OMUZ_PEN_ABDUKSIYON":
            feedback_talimat = "8. Yanda Sallama: One egilin, kolu saga sola sallayin."
            angle_sol = calculate_angle_3d(r_shoulder, l_shoulder, l_wrist)
            angle_sag = calculate_angle_3d(l_shoulder, r_shoulder, r_wrist)
            msg_sol = counter_pen_abduksiyon_sol.count(angle_sol)
            msg_sag = counter_pen_abduksiyon_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "OMUZ_CEMBER":
            feedback_talimat = "9. Cember Cizme (15sn): One egilin, kolunuzla cember cizin."
            
            is_active_sol = False
            is_active_sag = False
            
            # Sol kol hareketini algıla
            if last_wrist_pos_sol:
                distance_sol = calculate_distance_3d(last_wrist_pos_sol, l_wrist)
                if distance_sol > MOVEMENT_THRESHOLD:
                    is_active_sol = True
            last_wrist_pos_sol = l_wrist # Pozisyonu güncelle
            
            # Sağ kol hareketini algıla
            if last_wrist_pos_sag:
                distance_sag = calculate_distance_3d(last_wrist_pos_sag, r_wrist)
                if distance_sag > MOVEMENT_THRESHOLD:
                    is_active_sag = True
            last_wrist_pos_sag = r_wrist # Pozisyonu güncelle

            # Zamanlayıcıları güncelle
            msg_sol = timer_cember_sol.update_feedback(is_active_sol)
            msg_sag = timer_cember_sag.update_feedback(is_active_sag)
            
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        # --- 3. DUVAR EGZERSİZLERİ ---
        elif current_exercise == "OMUZ_DUVAR_YANA":
            feedback_talimat = "10. Duvara Yana Acma: Duvara yan durun, kolunuzu yana acin."
            angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_elbow)
            angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_elbow)
            msg_sol = counter_duvar_yana_sol.count(angle_sol)
            msg_sag = counter_duvar_yana_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "OMUZ_DUVAR_ONE":
            feedback_talimat = "11. Duvara One Itme: Duvara donun, kolunuzu one itin."
            angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_elbow)
            angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_elbow)
            msg_sol = counter_duvar_one_sol.count(angle_sol)
            msg_sag = counter_duvar_one_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "OMUZ_DUVAR_GERIYE":
            feedback_talimat = "12. Duvara Geriye Itme: Duvara arkanizi donun, geriye itin."
            angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_wrist)
            angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_wrist)
            msg_sol = counter_duvar_geriye_sol.count(angle_sol)
            msg_sag = counter_duvar_geriye_sag.count(angle_sag)
            feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        elif current_exercise == "OMUZ_DUVAR_DISA":
            feedback_talimat = "13. Duvara Disa Itme: Dirsek bukulu, duvara disa itin."
            elbow_angle_sol = calculate_angle_3d(l_shoulder, l_elbow, l_wrist)
            elbow_angle_sag = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)
            if (elbow_angle_sol < 70 or elbow_angle_sol > 110) or (elbow_angle_sag < 70 or elbow_angle_sag > 110):
                feedback_mesaj = "POZISYON HATALI: Dirsekleri 90 derece bukun!"
            else:
                angle_sol = calculate_angle_3d(l_hip, l_shoulder, l_wrist)
                angle_sag = calculate_angle_3d(r_hip, r_shoulder, r_wrist)
                msg_sol = counter_duvar_disa_sol.count(angle_sol)
                msg_sag = counter_duvar_disa_sag.count(angle_sag)
                feedback_mesaj = f"[SOL: {msg_sol}]  [SAG: {msg_sag}]"

        # --- 4. GERME (YENİ EKLENDİ) ---
        elif current_exercise == "OMUZ_GERME":
            feedback_talimat = "14. Germe (15sn): Kolunuzu gogsunuze cekin ve duz tutun."
            
            # Pozisyon kontrolü
            # 1. Kol düz mü? (Omuz-Dirsek-Bilek > 150 derece)
            # 2. Kol göğse çekili mi? (Dirsek-Omuz-DiğerOmuz < 40 derece)
            
            # Sol kol kontrolü
            is_active_sol = False; pose_feedback_sol = ""
            angle_kol_duz_sol = calculate_angle_3d(l_shoulder, l_elbow, l_wrist)
            angle_kol_cekili_sol = calculate_angle_3d(l_elbow, l_shoulder, r_shoulder)
            
            if angle_kol_duz_sol < 150: pose_feedback_sol = "Kolu Duz Tut"
            elif angle_kol_cekili_sol > 40: pose_feedback_sol = "Kolu Daha Cok Cek"
            else: is_active_sol = True; pose_feedback_sol = "POZISYONU KORU"
            
            # Sağ kol kontrolü
            is_active_sag = False; pose_feedback_sag = ""
            angle_kol_duz_sag = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)
            angle_kol_cekili_sag = calculate_angle_3d(r_elbow, r_shoulder, l_shoulder)
            
            if angle_kol_duz_sag < 150: pose_feedback_sag = "Kolu Duz Tut"
            elif angle_kol_cekili_sag > 40: pose_feedback_sag = "Kolu Daha Cok Cek"
            else: is_active_sag = True; pose_feedback_sag = "POZISYONU KORU"

            # Zamanlayıcıları güncelle
            msg_sol = timer_germe_sol.update_feedback(is_active_sol)
            msg_sag = timer_germe_sag.update_feedback(is_active_sag)
            
            feedback_mesaj = f"[SOL: {pose_feedback_sol} - {msg_sol}]  [SAG: {pose_feedback_sag} - {msg_sag}]"
        
    except Exception as e:
        feedback_talimat = "Kullanici algilanmadi veya hesaplama hatasi."
        feedback_mesaj = f"Hata: {e}"
        # Hata durumunda da zamanlayıcıları duraklat
        timer_cember_sol.update_feedback(False)
        timer_cember_sag.update_feedback(False)
        timer_germe_sol.update_feedback(False) # YENİ
        timer_germe_sag.update_feedback(False) # YENİ
        
    return feedback_talimat, feedback_mesaj