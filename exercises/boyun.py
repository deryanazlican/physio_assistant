# exercises/boyun.py
# GÜNCEL TAM KOD (V7 - V0 (Orijinal) Kod Temelli + V2/V3 Düzeltmeleri)

import mediapipe as mp
import time 
import numpy as np
from utils.angles import calculate_angle_3d, calculate_distance_3d
from utils.counter import RepCounter 
from utils.logger import log_exercise

mp_pose = mp.solutions.pose.PoseLandmark

# --- YARDIMCI FONKSİYONLAR ---
def get_landmark_coords(landmarks, landmark_name):
    lm = landmarks[landmark_name.value]
    if lm.visibility < 0.5: # Orijinal V0 eşiği
        raise ValueError(f"{landmark_name.name} gorunmuyor!")
    return [lm.x, lm.y, lm.z]

def get_midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2]

# --- SAYAÇLARI VE DURUMLARI OLUŞTUR ---\

# V0 Orijinal ROM Ayarları (Sadece "az sayıyor" dediğin için eşik 7'ye düşürüldü)
lateral_counter_sag = RepCounter("Yana Egilme", "Sag", threshold_angle=7, target_reps=5, neutral_threshold=2)
lateral_counter_sol = RepCounter("Yana Egilme", "Sol", threshold_angle=7, target_reps=5, neutral_threshold=2)
rotasyon_counter_sag = RepCounter("Donme", "Sag", threshold_angle=7, target_reps=5, neutral_threshold=2)
rotasyon_counter_sol = RepCounter("Donme", "Sol", threshold_angle=7, target_reps=5, neutral_threshold=2)

# V2 DÜZELTMESİ (ROM Öne/Arkaya - BU DOĞRUYDU, KORUNDU)
NEUTRAL_ANGLE_FLEX_EKST = 30 
fleksiyon_counter = RepCounter("ROM Fleksiyon", "On", threshold_angle=15, target_reps=5, neutral_threshold=5)
ekstansiyon_counter = RepCounter("ROM Ekstansiyon", "Arka", threshold_angle=15, target_reps=5, neutral_threshold=5)


IZO_SURE_HEDEF = 5; IZO_DINLENME_SURESI = 3; IZO_TEKRAR_SAYISI = 3 
izo_fleks_state = "beklemede"; izo_fleks_timer = 0; izo_fleks_rep = 0; izo_fleks_aci = 0.0 
izo_ekst_state = "beklemede"; izo_ekst_timer = 0; izo_ekst_rep = 0; izo_ekst_aci = 0.0 
izo_lat_sag_state = "beklemede"; izo_lat_sag_timer = 0; izo_lat_sag_rep = 0; izo_lat_sag_aci = 0.0 
izo_lat_sol_state = "beklemede"; izo_lat_sol_timer = 0; izo_lat_sol_rep = 0; izo_lat_sol_aci = 0.0 
izo_rot_sag_state = "beklemede"; izo_rot_sag_timer = 0; izo_rot_sag_rep = 0; izo_rot_sag_aci = 0.0
izo_rot_sol_state = "beklemede"; izo_rot_sol_timer = 0; izo_rot_sol_rep = 0; izo_rot_sol_aci = 0.0

# --- SAYAÇ SIFIRLAMA FONKSİYONU ---
def reset_boyun_counters():
    global izo_fleks_state, izo_fleks_rep, izo_ekst_state, izo_ekst_rep
    global izo_lat_sag_state, izo_lat_sag_rep, izo_lat_sol_state, izo_lat_sol_rep
    global izo_rot_sag_state, izo_rot_sag_rep, izo_rot_sol_state, izo_rot_sol_rep
    print("Tum sayaclar sifirlandi.")
    lateral_counter_sag.reset(); lateral_counter_sol.reset()
    rotasyon_counter_sag.reset(); rotasyon_counter_sol.reset()
    fleksiyon_counter.reset(); ekstansiyon_counter.reset()
    izo_fleks_state = "beklemede"; izo_fleks_rep = 0; izo_ekst_state = "beklemede"; izo_ekst_rep = 0
    izo_lat_sag_state = "beklemede"; izo_lat_sag_rep = 0; izo_lat_sol_state = "beklemede"; izo_lat_sol_rep = 0
    izo_rot_sag_state = "beklemede"; izo_rot_sag_rep = 0; izo_rot_sol_state = "beklemede"; izo_rot_sol_rep = 0

# --- "PRO" YÖNLENDİRİCİ FONKSİYON ---
def get_exercise_feedback(exercise_name, landmarks):
    talimat = ""
    mesaj = ""
    try:
        if exercise_name == "MENU":
            talimat = "Lutfen bir hareket secin:"
        elif exercise_name == "ROM_LAT":
            talimat = "Basini YANA EG (Hedef: 7 der) ve merkeze don."
            mesaj = check_boyun_lateral_fleksiyon(landmarks)
        elif exercise_name == "ROM_ROT":
            talimat = "Basini YANA DONDUR (Hedef: 7 der) ve merkeze don."
            mesaj = check_boyun_rotasyon(landmarks)
        elif exercise_name == "ROM_FLEKS":
            talimat = "Basini ONE/ARKAYA EG (Hedef: 15 der) ve merkeze don."
            mesaj = check_boyun_fleksiyon_ekstansiyon(landmarks)
        elif exercise_name == "IZO_FLEKS":
            talimat = "Elini alnina koy ve 5sn IT (Kafani oynatma!)"
            mesaj = check_boyun_izometrik_fleksiyon(landmarks)
        elif exercise_name == "IZO_EKST": 
            talimat = "Elini basinin arkasina koy ve 5sn IT"
            mesaj = check_boyun_izometrik_ekstansiyon(landmarks)
        elif exercise_name == "IZO_LAT":
            talimat = "Elini sakagina koy ve 5sn IT (Kafani egme!)"
            mesaj = check_boyun_izometrik_lateral(landmarks)
        elif exercise_name == "IZO_ROT": 
            talimat = "Elini cenene koy ve 5sn DONDUR (Kafani dondurme!)"
            mesaj = check_boyun_izometrik_rotasyon(landmarks)
    except Exception as e:
        talimat = "Lutfen kamerada tam gorunun."
        mesaj = f"Algilama hatasi: {e}"
    return talimat, mesaj

# --- ROM HAREKETLERİ ---

# HATA 1: ROM ÖNE/ARKAYA (V2'den beri DÜZGÜN ÇALIŞIYOR)
def check_boyun_fleksiyon_ekstansiyon(landmarks):
    global NEUTRAL_ANGLE_FLEX_EKST
    sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
    bas_ortasi = get_midpoint(sol_kulak, sag_kulak); sol_omuz = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER)
    sag_omuz = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER); omuz_ortasi = get_midpoint(sol_omuz, sag_omuz)
    p_forward = np.array(omuz_ortasi) + np.array([0, 0, -0.1])
    aci = calculate_angle_3d(bas_ortasi, omuz_ortasi, p_forward)
    mutlak_aci = int(90 - aci)
    relative_angle_on = max(0, mutlak_aci - NEUTRAL_ANGLE_FLEX_EKST)
    relative_angle_arka = max(0, NEUTRAL_ANGLE_FLEX_EKST - mutlak_aci)
    rep_mesaji_on = fleksiyon_counter.count(relative_angle_on)
    rep_mesaji_arka = ekstansiyon_counter.count(relative_angle_arka)
    return f"Aci: {mutlak_aci} der | {rep_mesaji_on} | {rep_mesaji_arka}"

# Hata "Az Sayıyor" (Eşik 7'ye düşürüldü)
def check_boyun_lateral_fleksiyon(landmarks):
    sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
    bas_ortasi = get_midpoint(sol_kulak, sag_kulak); sol_omuz = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER)
    sag_omuz = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER); omuz_ortasi = get_midpoint(sol_omuz, sag_omuz)
    p_horizontal = np.array(omuz_ortasi) + np.array([0.1, 0, 0]); aci = calculate_angle_3d(bas_ortasi, omuz_ortasi, p_horizontal)
    aci_gosterge = int(90 - aci); yon_threshold = 5; yon = "Orta"
    if aci_gosterge > yon_threshold: yon = "Saga"
    elif aci_gosterge < -yon_threshold: yon = "Sola"
    if yon == "Saga":
        rep_mesaji_sag = lateral_counter_sag.count(abs(aci_gosterge)); rep_mesaji_sol = lateral_counter_sol.get_current_message() 
    elif yon == "Sola":
        rep_mesaji_sag = lateral_counter_sag.get_current_message(); rep_mesaji_sol = lateral_counter_sol.count(abs(aci_gosterge))
    else: 
        rep_mesaji_sag = lateral_counter_sag.count(abs(aci_gosterge)); rep_mesaji_sol = lateral_counter_sol.count(abs(aci_gosterge))
    mesaj = f"Aci: {abs(aci_gosterge)} der | {rep_mesaji_sag} | {rep_mesaji_sol}" 
    omuz_farki_y = abs(sol_omuz[1] - sag_omuz[1])
    if omuz_farki_y > 0.05: mesaj = "HILE: Omuzlarini kaldirma!"
    return mesaj

# HATA 2: ROM DÖNME (V3'te DÜZELMİŞTİ)
def check_boyun_rotasyon(landmarks):
    sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
    burun = get_landmark_coords(landmarks, mp_pose.NOSE); head_width = sag_kulak[0] - sol_kulak[0]
    # V3 Düzeltmesi: Profil kontrolü kaldırıldı
    nose_normalized_pos = (burun[0] - sol_kulak[0]) / head_width; rotation_normalized = (nose_normalized_pos - 0.5) * 2
    aci = int(rotation_normalized * 80); yon_threshold = 5; yon = "Orta"
    if aci > yon_threshold: yon = "Saga"
    elif aci < -yon_threshold: yon = "Sola"
    if yon == "Saga":
        rep_mesaji_sag = rotasyon_counter_sag.count(abs(aci)); rep_mesaji_sol = rotasyon_counter_sol.get_current_message()
    elif yon == "Sola":
        rep_mesaji_sag = rotasyon_counter_sag.get_current_message(); rep_mesaji_sol = rotasyon_counter_sol.count(abs(aci))
    else: 
        rep_mesaji_sag = rotasyon_counter_sag.count(abs(aci)); rep_mesaji_sol = rotasyon_counter_sol.count(abs(aci))
    return f"Aci: {abs(aci)} der | {rep_mesaji_sag} | {rep_mesaji_sol}"

# --- İZOMETRİK HAREKETLER ---

# HATA 3: İZO ÖNE (V7 - Orijinal V0 Mantığına Geri Dönüldü)
def check_boyun_izometrik_fleksiyon(landmarks):
    global izo_fleks_state, izo_fleks_timer, izo_fleks_rep, izo_fleks_aci
    el_pozisyonda = False
    try:
        # V0 (Orijinal) PARMAK (INDEX) kullanılıyor
        sol_parmak = get_landmark_coords(landmarks, mp_pose.LEFT_INDEX); sag_parmak = get_landmark_coords(landmarks, mp_pose.RIGHT_INDEX)
        burun_noktasi = get_landmark_coords(landmarks, mp_pose.NOSE); mesafe_sol = calculate_distance_3d(sol_parmak, burun_noktasi)
        mesafe_sag = calculate_distance_3d(sag_parmak, burun_noktasi); sol_parmak_y = sol_parmak[1]; burun_y = burun_noktasi[1]; sag_parmak_y = sag_parmak[1]
        
        # V0 (Orijinal) Eşik (0.15) ve Y kontrolü
        if (mesafe_sol < 0.15 and sol_parmak_y < burun_y) or (mesafe_sag < 0.15 and sag_parmak_y < burun_y):
            el_pozisyonda = True
    except Exception as e: el_pozisyonda = False 
    
    # --- Sayaç mantığı (dokunulmadı) ---
    sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sol_omuz = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER)
    sag_omuz = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER); fleksiyon_aci = calculate_angle_3d(sol_kulak, sol_omuz, sag_omuz)
    if izo_fleks_rep >= IZO_TEKRAR_SAYISI:
        if izo_fleks_state != "tamamlandi": log_exercise("Izometrik Fleksiyon", IZO_TEKRAR_SAYISI, "On"); izo_fleks_state = "tamamlandi"
        return f"TAMAMLANDI!"
    if izo_fleks_state == "beklemede":
        if el_pozisyonda:
            izo_fleks_state = "sayimda"; izo_fleks_timer = time.time(); izo_fleks_aci = fleksiyon_aci 
            return f"TUT! ({izo_fleks_rep + 1}/{IZO_TEKRAR_SAYISI})"
        else: return f"Tekrar: {izo_fleks_rep}/{IZO_TEKRAR_SAYISI}"
    elif izo_fleks_state == "sayimda":
        gecen_saniye = time.time() - izo_fleks_timer; aci_farki = abs(fleksiyon_aci - izo_fleks_aci); bas_dik = (aci_farki < 15)
        if not el_pozisyonda: izo_fleks_state = "beklemede"; return f"HATA: Elini indirdin!"
        if not bas_dik: izo_fleks_state = "beklemede"; return f"HILE: Kafani oynattin!"
        if gecen_saniye >= IZO_SURE_HEDEF:
            izo_fleks_rep += 1; izo_fleks_state = "dinlen"; izo_fleks_timer = time.time(); return f"HARIKA! ({izo_fleks_rep}/{IZO_TEKRAR_SAYISI})"
        else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"TUT! {kalan_saniye}s... ({izo_fleks_rep + 1}/{IZO_TEKRAR_SAYISI})"
    elif izo_fleks_state == "dinlen":
        if time.time() - izo_fleks_timer >= IZO_DINLENME_SURESI:
            izo_fleks_state = "beklemede"; return f"Hazir ol... ({izo_fleks_rep + 1}/{IZO_TEKRAR_SAYISI})"
        else: return f"Dinlen... ({izo_fleks_rep}/{IZO_TEKRAR_SAYISI})"

# HATA 3: İZO ARKAYA (V7 - Orijinal V0 Mantığına Geri Dönüldü)
def check_boyun_izometrik_ekstansiyon(landmarks):
    global izo_ekst_state, izo_ekst_timer, izo_ekst_rep, izo_ekst_aci
    el_pozisyonda = False
    try:
        # V0 (Orijinal) PARMAK (INDEX) kullanılıyor
        sol_parmak = get_landmark_coords(landmarks, mp_pose.LEFT_INDEX); sag_parmak = get_landmark_coords(landmarks, mp_pose.RIGHT_INDEX)
        sol_kulak_nokta = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak_nokta = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
        mesafe_sol = calculate_distance_3d(sol_parmak, sol_kulak_nokta); mesafe_sag = calculate_distance_3d(sag_parmak, sag_kulak_nokta)
        
        # V0 (Orijinal) Eşik (0.55). Görüntüdeki (V6) hatayı çözmek için bunu 0.25'e düşürüyoruz.
        if (mesafe_sol < 0.25 or mesafe_sag < 0.25): el_pozisyonda = True 
    except Exception as e: el_pozisyonda = False 
    
    # --- Sayaç mantığı (dokunulmadı) ---
    sol_kulak_aci = get_landmark_coords(landmarks, mp_pose.LEFT_EAR)
    sol_omuz = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER); sag_omuz = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER)
    fleksiyon_aci = calculate_angle_3d(sol_kulak_aci, sol_omuz, sag_omuz)
    if izo_ekst_rep >= IZO_TEKRAR_SAYISI:
        if izo_ekst_state != "tamamlandi": log_exercise("Izometrik Ekstansiyon", IZO_TEKRAR_SAYISI, "Arka"); izo_ekst_state = "tamamlandi"
        return f"TAMAMLANDI!"
    if izo_ekst_state == "beklemede":
        if el_pozisyonda:
            izo_ekst_state = "sayimda"; izo_ekst_timer = time.time(); izo_ekst_aci = fleksiyon_aci 
            return f"TUT! ({izo_ekst_rep + 1}/{IZO_TEKRAR_SAYISI})"
        else: return f"Tekrar: {izo_ekst_rep}/{IZO_TEKRAR_SAYISI}"
    elif izo_ekst_state == "sayimda":
        gecen_saniye = time.time() - izo_ekst_timer; aci_farki = abs(fleksiyon_aci - izo_ekst_aci); bas_dik = (aci_farki < 15) 
        if not el_pozisyonda: izo_ekst_state = "beklemede"; return f"HATA: Elini indirdin!"
        if not bas_dik: izo_ekst_state = "beklemede"; return f"HILE: Kafani oynattin!"
        if gecen_saniye >= IZO_SURE_HEDEF:
            izo_ekst_rep += 1; izo_ekst_state = "dinlen"; izo_ekst_timer = time.time(); return f"HARIKA! ({izo_ekst_rep}/{IZO_TEKRAR_SAYISI})"
        else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"TUT! {kalan_saniye}s... ({izo_ekst_rep + 1}/{IZO_TEKRAR_SAYISI})"
    elif izo_ekst_state == "dinlen":
        if time.time() - izo_ekst_timer >= IZO_DINLENME_SURESI:
            izo_ekst_state = "beklemede"; return f"Hazir ol... ({izo_ekst_rep + 1}/{IZO_TEKRAR_SAYISI})"
        else: return f"HARIKA! ({izo_ekst_rep}/{IZO_TEKRAR_SAYISI})"

# HATA 4B: İZO YANA (V7 - Orijinal V0 Mantığına Geri Dönüldü)
def check_boyun_izometrik_lateral(landmarks):
    global izo_lat_sag_state, izo_lat_sag_timer, izo_lat_sag_rep, izo_lat_sag_aci
    global izo_lat_sol_state, izo_lat_sol_timer, izo_lat_sol_rep, izo_lat_sol_aci
    sag_el_pozisyonda = False; sol_el_pozisyonda = False
    
    # V0 (Orijinal) Mantık: Önce SAĞ eli (RIGHT_INDEX) ara
    # V6'daki hatanın (diğer elin çökmesi) yaşanmaması için ayrı try-except blokları
    try:
        sag_parmak = get_landmark_coords(landmarks, mp_pose.RIGHT_INDEX); sag_kulak_nokta = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
        mesafe_sag = calculate_distance_3d(sag_parmak, sag_kulak_nokta)
        # Eşiği 0.25'e düşürüyoruz
        if mesafe_sag < 0.25: sag_el_pozisyonda = True 
    except Exception as e: sag_el_pozisyonda = False
    
    # V0 (Orijinal) Mantık: Sonra SOL eli (LEFT_INDEX) ara
    try:
        sol_parmak = get_landmark_coords(landmarks, mp_pose.LEFT_INDEX); sol_kulak_nokta = get_landmark_coords(landmarks, mp_pose.LEFT_EAR)
        mesafe_sol = calculate_distance_3d(sol_parmak, sol_kulak_nokta)
        # Eşiği 0.25'e düşürüyoruz
        if mesafe_sol < 0.25: sol_el_pozisyonda = True 
    except Exception as e: sol_el_pozisyonda = False
    
    sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
    bas_ortasi = get_midpoint(sol_kulak, sag_kulak); sol_omuz = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER)
    sag_omuz = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER); omuz_ortasi = get_midpoint(sol_omuz, sag_omuz)
    p_horizontal = np.array(omuz_ortasi) + np.array([0.1, 0, 0])
    lateral_aci = int(90 - calculate_angle_3d(bas_ortasi, omuz_ortasi, p_horizontal))
    
    # V0 (Orijinal) ve V3 Mantığı: Sağ/Sol etiketleri TERS DEĞİL.
    # main.py'deki flip'e göre (Gerçek Sol El = RIGHT_WRIST = Ekranda Sağ)
    
    if sag_el_pozisyonda and not sol_el_pozisyonda: # Sağ el (Gerçek Sol, Ekranda Sağ)
        if izo_lat_sag_rep >= IZO_TEKRAR_SAYISI:
            if izo_lat_sag_state != "tamamlandi": log_exercise("Izometrik Lateral", IZO_TEKRAR_SAYISI, "Sag"); izo_lat_sag_state = "tamamlandi"
            return f"Sag: TAMAMLANDI!"
        if izo_lat_sag_state == "beklemede":
            izo_lat_sag_state = "sayimda"; izo_lat_sag_timer = time.time(); izo_lat_sag_aci = lateral_aci 
            return f"Sag: TUT! ({izo_lat_sag_rep + 1}/{IZO_TEKRAR_SAYISI})"
        elif izo_lat_sag_state == "sayimda":
            gecen_saniye = time.time() - izo_lat_sag_timer; aci_farki = abs(lateral_aci - izo_lat_sag_aci); bas_dik = (aci_farki < 10 or lateral_aci > 0) 
            if not sag_el_pozisyonda: izo_lat_sag_state = "beklemede"; return f"HATA (Sag): Elini indirdin!"
            if not bas_dik: izo_lat_sag_state = "beklemede"; return f"HILE (Sag): Kafani yana egme!"
            if gecen_saniye >= IZO_SURE_HEDEF:
                izo_lat_sag_rep += 1; izo_lat_sag_state = "dinlen"; izo_lat_sag_timer = time.time(); return f"HARIKA! (Sag) ({izo_lat_sag_rep}/{IZO_TEKRAR_SAYISI})"
            else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Sag: TUT! {kalan_saniye}s... ({izo_lat_sag_rep + 1}/{IZO_TEKRAR_SAYISI})"
        elif izo_lat_sag_state == "dinlen":
            if time.time() - izo_lat_sag_timer >= IZO_DINLENME_SURESI:
                izo_lat_sag_state = "beklemede"; return f"Hazir ol (Sag)... ({izo_lat_sag_rep + 1}/{IZO_TEKRAR_SAYISI})"
            else: return f"Dinlen (Sag)... ({izo_lat_sag_rep}/{IZO_TEKRAR_SAYISI})"
    
    elif sol_el_pozisyonda and not sag_el_pozisyonda: # Sol el (Gerçek Sağ, Ekranda Sol)
        if izo_lat_sol_rep >= IZO_TEKRAR_SAYISI:
            if izo_lat_sol_state != "tamamlandi": log_exercise("Izometrik Lateral", IZO_TEKRAR_SAYISI, "Sol"); izo_lat_sol_state = "tamamlandi"
            return f"Sol: TAMAMLANDI!"
        if izo_lat_sol_state == "beklemede":
            izo_lat_sol_state = "sayimda"; izo_lat_sol_timer = time.time(); izo_lat_sol_aci = lateral_aci 
            return f"Sol: TUT! ({izo_lat_sol_rep + 1}/{IZO_TEKRAR_SAYISI})"
        elif izo_lat_sol_state == "sayimda":
            gecen_saniye = time.time() - izo_lat_sol_timer; aci_farki = abs(lateral_aci - izo_lat_sol_aci); bas_dik = (aci_farki < 10 or lateral_aci < 0) 
            if not sol_el_pozisyonda: izo_lat_sol_state = "beklemede"; return f"HATA (Sol): Elini indirdin!"
            if not bas_dik: izo_lat_sol_state = "beklemede"; return f"HILE (Sol): Kafani yana egme!"
            if gecen_saniye >= IZO_SURE_HEDEF:
                izo_lat_sol_rep += 1; izo_lat_sol_state = "dinlen"; izo_lat_sol_timer = time.time(); return f"HARIKA! (Sol) ({izo_lat_sol_rep}/{IZO_TEKRAR_SAYISI})"
            else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Sol: TUT! {kalan_saniye}s... ({izo_lat_sol_rep + 1}/{IZO_TEKRAR_SAYISI})"
        elif izo_lat_sol_state == "dinlen":
            if time.time() - izo_lat_sol_timer >= IZO_DINLENME_SURESI:
                izo_lat_sol_state = "beklemede"; return f"Hazir ol (Sol)... ({izo_lat_sol_rep + 1}/{IZO_TEKRAR_SAYISI})"
            else: return f"Dinlen (Sol)... ({izo_lat_sol_rep}/{IZO_TEKRAR_SAYISI})"
    else:
        return f"(Sag: {izo_lat_sag_rep}/{IZO_TEKRAR_SAYISI} | Sol: {izo_lat_sol_rep}/{IZO_TEKRAR_SAYISI})"

# HATA 5: İZO DÖNDÜRME (V7 - Orijinal V0 Mantığına Geri Dönüldü)
def check_boyun_izometrik_rotasyon(landmarks):
    global izo_rot_sag_state, izo_rot_sag_timer, izo_rot_sag_rep, izo_rot_sag_aci
    global izo_rot_sol_state, izo_rot_sol_timer, izo_rot_sol_rep, izo_rot_sol_aci
    sag_el_pozisyonda = False; sol_el_pozisyonda = False

    # V0 (Orijinal) Mantık: Önce SAĞ eli (RIGHT_INDEX) ara
    try:
        agiz_sol = get_landmark_coords(landmarks, mp_pose.MOUTH_LEFT); agiz_sag = get_landmark_coords(landmarks, mp_pose.MOUTH_RIGHT)
        agiz_ortasi = get_midpoint(agiz_sol, agiz_sag)
        sag_parmak = get_landmark_coords(landmarks, mp_pose.RIGHT_INDEX)
        mesafe_sag = calculate_distance_3d(sag_parmak, agiz_ortasi)
        if mesafe_sag < 0.15: sag_el_pozisyonda = True # Orijinal V0 eşiği
    except Exception as e: sag_el_pozisyonda = False
    
    # V0 (Orijinal) Mantık: Sonra SOL eli (LEFT_INDEX) ara
    try:
        agiz_sol = get_landmark_coords(landmarks, mp_pose.MOUTH_LEFT); agiz_sag = get_landmark_coords(landmarks, mp_pose.MOUTH_RIGHT)
        agiz_ortasi = get_midpoint(agiz_sol, agiz_sag)
        sol_parmak = get_landmark_coords(landmarks, mp_pose.LEFT_INDEX)
        mesafe_sol = calculate_distance_3d(sol_parmak, agiz_ortasi)
        if mesafe_sol < 0.15: sol_el_pozisyonda = True # Orijinal V0 eşiği
    except Exception as e: sol_el_pozisyonda = False

    sol_kulak_aci = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak_aci = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
    burun = get_landmark_coords(landmarks, mp_pose.NOSE); head_width = sag_kulak_aci[0] - sol_kulak_aci[0]
    
    # V3 DÜZELTMESİ (Hata 5): "Algılanma hatası". Eşik 0.07'ye ayarlandı.
    if head_width < 0.07: raise Exception("HILE: Kafani dondurme!") 
    
    nose_normalized_pos = (burun[0] - sol_kulak_aci[0]) / head_width
    rotasyon_aci = int(((nose_normalized_pos - 0.5) * 2) * 80)
    
    # V0 (Orijinal) ve V3 Mantığı: Sağ/Sol etiketleri TERS DEĞİL.
    
    if sag_el_pozisyonda and not sol_el_pozisyonda: # Sağ el (Gerçek Sol, Ekranda Sağ)
        if izo_rot_sag_rep >= IZO_TEKRAR_SAYISI: 
            if izo_rot_sag_state != "tamamlandi": log_exercise("Izometrik Rotasyon", IZO_TEKRAR_SAYISI, "Sag"); izo_rot_sag_state = "tamamlandi"
            return f"Sag Donus: TAMAMLANDI!"
        if izo_rot_sag_state == "beklemede":
            izo_rot_sag_state = "sayimda"; izo_rot_sag_timer = time.time(); izo_rot_sag_aci = rotasyon_aci 
            return f"Sag Donus: TUT! ({izo_rot_sag_rep + 1}/{IZO_TEKRAR_SAYISI})"
        elif izo_rot_sag_state == "sayimda":
            gecen_saniye = time.time() - izo_rot_sag_timer; aci_farki = abs(rotasyon_aci - izo_rot_sag_aci); bas_dondu = (aci_farki > 15 or rotasyon_aci > 0) 
            if not sag_el_pozisyonda: izo_rot_sag_state = "beklemede"; return f"HATA (Sag Donus): Elini indirdin!"
            if bas_dondu: izo_rot_sag_state = "beklemede"; return f"HILE (Sag Donus): Kafani dondurme!"
            if gecen_saniye >= IZO_SURE_HEDEF:
                izo_rot_sag_rep += 1; izo_rot_sag_state = "dinlen"; izo_rot_sag_timer = time.time(); return f"HARIKA! (Sag Donus) ({izo_rot_sag_rep}/{IZO_TEKRAR_SAYISI})"
            else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Sag Donus: TUT! {kalan_saniye}s... ({izo_rot_sag_rep + 1}/{IZO_TEKRAR_SAYISI})"
        elif izo_rot_sag_state == "dinlen":
            if time.time() - izo_rot_sag_timer >= IZO_DINLENME_SURESI:
                izo_rot_sag_state = "beklemede"; return f"Hazir ol (Sag Donus)... ({izo_rot_sag_rep + 1}/{IZO_TEKRAR_SAYISI})"
            else: return f"HARIKA! (Sag Donus) ({izo_rot_sag_rep}/{IZO_TEKRAR_SAYISI})"

    elif sol_el_pozisyonda and not sag_el_pozisyonda: # Sol el (Gerçek Sağ, Ekranda Sol)
        if izo_rot_sol_rep >= IZO_TEKRAR_SAYISI: 
            if izo_rot_sol_state != "tamamlandi": log_exercise("Izometrik Rotasyon", IZO_TEKRAR_SAYISI, "Sol"); izo_rot_sol_state = "tamamlandi"
            return f"Sol Donus: TAMAMLANDI!"
        if izo_rot_sol_state == "beklemede":
            izo_rot_sol_state = "sayimda"; izo_rot_sol_timer = time.time(); izo_rot_sol_aci = rotasyon_aci 
            return f"Sol Donus: TUT! ({izo_rot_sol_rep + 1}/{IZO_TEKRAR_SAYISI})"
        elif izo_rot_sol_state == "sayimda":
            gecen_saniye = time.time() - izo_rot_sol_timer; aci_farki = abs(rotasyon_aci - izo_rot_sol_aci); bas_dondu = (aci_farki > 15 or rotasyon_aci < 0) 
            if not sol_el_pozisyonda: izo_rot_sol_state = "beklemede"; return f"HATA (Sol Donus): Elini indirdin!"
            if bas_dondu: izo_rot_sol_state = "beklemede"; return f"HILE (Sol Donus): Kafani dondurme!"
            if gecen_saniye >= IZO_SURE_HEDEF:
                izo_rot_sol_rep += 1; izo_rot_sol_state = "dinlen"; izo_rot_sol_timer = time.time(); return f"HARIKA! (Sol Donus) ({izo_rot_sol_rep}/{IZO_TEKRAR_SAYISI})"
            else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Sol Donus: TUT! {kalan_saniye}s... ({izo_rot_sol_rep + 1}/{IZO_TEKRAR_SAYISI})"
        elif izo_rot_sol_state == "dinlen":
            if time.time() - izo_rot_sol_timer >= IZO_DINLENME_SURESI:
                izo_rot_sol_state = "beklemede"; return f"Hazir ol (Sol Donus)... ({izo_rot_sol_rep + 1}/{IZO_TEKRAR_SAYISI})"
            else: return f"HARIKA! (Sol Donus) ({izo_rot_sol_rep}/{IZO_TEKRAR_SAYISI})"
    else:
        return f"(Sag: {izo_rot_sag_rep}/{IZO_TEKRAR_SAYISI} | Sol: {izo_rot_sol_rep}/{IZO_TEKRAR_SAYISI})"