# exercises/boyun.py
# GÜNCEL TAM KOD (O REZİL "OLCULUYOR..." HATASI DÜZELTİLDİ)
# Artık izometrik fonksiyonlar el görünmezse çökmeyecek.

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
    return [lm.x, lm.y, lm.z]

def get_midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2]

# --- SAYAÇLARI VE DURUMLARI OLUŞTUR ---
lateral_counter_sag = RepCounter("Yana Egilme", "Sag", threshold_angle=10, target_reps=5, neutral_threshold=2)
lateral_counter_sol = RepCounter("Yana Egilme", "Sol", threshold_angle=10, target_reps=5, neutral_threshold=2)
rotasyon_counter_sag = RepCounter("Donme", "Sag", threshold_angle=10, target_reps=5, neutral_threshold=2)
rotasyon_counter_sol = RepCounter("Donme", "Sol", threshold_angle=10, target_reps=5, neutral_threshold=2)
fleksiyon_counter = RepCounter("ROM Fleksiyon", "On", threshold_angle=10, target_reps=5, neutral_threshold=2)
ekstansiyon_counter = RepCounter("ROM Ekstansiyon", "Arka", threshold_angle=10, target_reps=5, neutral_threshold=2)

IZO_SURE_HEDEF = 5; IZO_DINLENME_SURESI = 3; IZO_TEKRAR_SAYISI = 3 
izo_fleks_state = "beklemede"; izo_fleks_timer = 0; izo_fleks_rep = 0; izo_fleks_aci = 0.0 
izo_ekst_state = "beklemede"; izo_ekst_timer = 0; izo_ekst_rep = 0; izo_ekst_aci = 0.0 
izo_lat_sag_state = "beklemede"; izo_lat_sag_timer = 0; izo_lat_sag_rep = 0; izo_lat_sag_aci = 0.0 
izo_lat_sol_state = "beklemede"; izo_lat_sol_timer = 0; izo_lat_sol_rep = 0; izo_lat_sol_aci = 0.0 
izo_rot_sag_state = "beklemede"; izo_rot_sag_timer = 0; izo_rot_sag_rep = 0; izo_rot_sag_aci = 0.0
izo_rot_sol_state = "beklemede"; izo_rot_sol_timer = 0; izo_rot_sol_rep = 0; izo_rot_sol_aci = 0.0

# --- SAYAÇ SIFIRLAMA FONKSİYONU ---
def reset_boyun_counters():
    """ Menüye dönüldüğünde tüm sayaçları sıfırlar. """
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

# --- ROM HAREKETLERİ (Aynı kaldı) ---
def check_boyun_fleksiyon_ekstansiyon(landmarks):
    try:
        sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sol_omuz = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER)
        sag_omuz = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER); aci = calculate_angle_3d(sol_kulak, sol_omuz, sag_omuz)
        aci_gosterge = int(90 - aci); yon_threshold = 5; yon = "Orta"
        if aci_gosterge > yon_threshold: yon = "One"
        elif aci_gosterge < -yon_threshold: yon = "Arka"
        if yon == "One":
            rep_mesaji_on = fleksiyon_counter.count(abs(aci_gosterge)); rep_mesaji_arka = ekstansiyon_counter.get_current_message()
        elif yon == "Arka":
            rep_mesaji_on = fleksiyon_counter.get_current_message(); rep_mesaji_arka = ekstansiyon_counter.count(abs(aci_gosterge))
        else: 
            rep_mesaji_on = fleksiyon_counter.count(abs(aci_gosterge)); rep_mesaji_arka = ekstansiyon_counter.count(abs(aci_gosterge))
        return f"Fleks/Eks: {aci_gosterge} derece | {rep_mesaji_on} | {rep_mesaji_arka}"
    except Exception as e: return "Pozisyon aliniyor..."

def check_boyun_lateral_fleksiyon(landmarks):
    try:
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
        mesaj = f"Yana Egilme: {abs(aci_gosterge)} derece | {rep_mesaji_sag} | {rep_mesaji_sol}" 
        omuz_farki_y = abs(sol_omuz[1] - sag_omuz[1])
        if omuz_farki_y > 0.05: mesaj = "HILE: Omuzlarini kaldirma!"
        return mesaj
    except Exception as e: return "Pozisyon aliniyor..."

def check_boyun_rotasyon(landmarks):
    try:
        sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
        burun = get_landmark_coords(landmarks, mp_pose.NOSE); head_width = sag_kulak[0] - sol_kulak[0]
        if head_width < 0.005: return "Rotasyon: Profil algilandi"
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
        return f"Rotasyon: {abs(aci)} derece | {rep_mesaji_sag} | {rep_mesaji_sol}"
    except Exception as e: return "Rotasyon olculuyor..."

# --- İZOMETRİK HAREKETLER (ARTIK GÜVENLİ) ---

# Hareket 1 (Öne)
def check_boyun_izometrik_fleksiyon(landmarks):
    global izo_fleks_state, izo_fleks_timer, izo_fleks_rep, izo_fleks_aci
    
    # --- "PRO" GÜVENLİ POZİSYON KONTROLÜ ---
    el_pozisyonda = False
    try:
        sol_parmak = get_landmark_coords(landmarks, mp_pose.LEFT_INDEX); sag_parmak = get_landmark_coords(landmarks, mp_pose.RIGHT_INDEX)
        burun_noktasi = get_landmark_coords(landmarks, mp_pose.NOSE); mesafe_sol = calculate_distance_3d(sol_parmak, burun_noktasi)
        mesafe_sag = calculate_distance_3d(sag_parmak, burun_noktasi); sol_parmak_y = sol_parmak[1]; burun_y = burun_noktasi[1]; sag_parmak_y = sag_parmak[1]
        if (mesafe_sol < 0.2 and sol_parmak_y < burun_y) or (mesafe_sag < 0.2 and sag_parmak_y < burun_y):
            el_pozisyonda = True
    except Exception as e:
        el_pozisyonda = False # El görünmüyorsa, pozisyonda değildir. Çökme.
    # --- GÜVENLİ KONTROL SONU ---

    try:
        sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sol_omuz = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER)
        sag_omuz = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER); fleksiyon_aci = calculate_angle_3d(sol_kulak, sol_omuz, sag_omuz)
        
        # STATE MACHINE...
        if izo_fleks_rep >= IZO_TEKRAR_SAYISI:
            if izo_fleks_state != "tamamlandi": log_exercise("Izometrik Fleksiyon", IZO_TEKRAR_SAYISI, "On"); izo_fleks_state = "tamamlandi"
            return f"Izometrik (One): TAMAMLANDI!"
        if izo_fleks_state == "beklemede":
            if el_pozisyonda:
                izo_fleks_state = "sayimda"; izo_fleks_timer = time.time(); izo_fleks_aci = fleksiyon_aci 
                return f"Izometrik ({izo_fleks_rep + 1}/{IZO_TEKRAR_SAYISI}): {IZO_SURE_HEDEF}s TUT!"
            else: return f"Izometrik ({izo_fleks_rep}/{IZO_TEKRAR_SAYISI}): Elini alnina koy"
        elif izo_fleks_state == "sayimda":
            gecen_saniye = time.time() - izo_fleks_timer; aci_farki = abs(fleksiyon_aci - izo_fleks_aci); bas_dik = (aci_farki < 15)
            if not el_pozisyonda: izo_fleks_state = "beklemede"; return f"HATA ({izo_fleks_rep}/{IZO_TEKRAR_SAYISI}): Elini indirdin!"
            if not bas_dik: izo_fleks_state = "beklemede"; return f"HILE ({izo_fleks_rep}/{IZO_TEKRAR_SAYISI}): Kafani oynattin!"
            if gecen_saniye >= IZO_SURE_HEDEF:
                izo_fleks_rep += 1; izo_fleks_state = "dinlen"; izo_fleks_timer = time.time(); return f"HARIKA! ({izo_fleks_rep}/{IZO_TEKRAR_SAYISI}). Dinlen."
            else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Izometrik ({izo_fleks_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT! {kalan_saniye}s..."
        elif izo_fleks_state == "dinlen":
            if time.time() - izo_fleks_timer >= IZO_DINLENME_SURESI:
                izo_fleks_state = "beklemede"; return f"Hazir ol ({izo_fleks_rep + 1}/{IZO_TEKRAR_SAYISI})... Elini alnina koy."
            else: return f"HARIKA! ({izo_fleks_rep}/{IZO_TEKRAR_SAYISI}). Dinlen."
    except Exception as e: return f"Izometrik ({izo_fleks_rep}/{IZO_TEKRAR_SAYISI}): Yuzunu goster"

# Hareket 2 (Arkaya)
def check_boyun_izometrik_ekstansiyon(landmarks):
    global izo_ekst_state, izo_ekst_timer, izo_ekst_rep, izo_ekst_aci
    
    el_pozisyonda = False
    try:
        sol_parmak = get_landmark_coords(landmarks, mp_pose.LEFT_INDEX); sag_parmak = get_landmark_coords(landmarks, mp_pose.RIGHT_INDEX)
        sol_kulak_nokta = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak_nokta = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
        mesafe_sol = calculate_distance_3d(sol_parmak, sol_kulak_nokta); mesafe_sag = calculate_distance_3d(sag_parmak, sag_kulak_nokta)
        if (mesafe_sol < 0.55 or mesafe_sag < 0.55):
            el_pozisyonda = True
    except Exception as e:
        el_pozisyonda = False 

    try:
        sol_kulak_aci = get_landmark_coords(landmarks, mp_pose.LEFT_EAR)
        sol_omuz = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER); sag_omuz = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER)
        fleksiyon_aci = calculate_angle_3d(sol_kulak_aci, sol_omuz, sag_omuz)
        
        if izo_ekst_rep >= IZO_TEKRAR_SAYISI:
            if izo_ekst_state != "tamamlandi": log_exercise("Izometrik Ekstansiyon", IZO_TEKRAR_SAYISI, "Arka"); izo_ekst_state = "tamamlandi"
            return f"Izometrik (Arka): TAMAMLANDI!"
        if izo_ekst_state == "beklemede":
            if el_pozisyonda:
                izo_ekst_state = "sayimda"; izo_ekst_timer = time.time(); izo_ekst_aci = fleksiyon_aci 
                return f"Izometrik ({izo_ekst_rep + 1}/{IZO_TEKRAR_SAYISI}): {IZO_SURE_HEDEF}s TUT!"
            else: return f"Izometrik ({izo_ekst_rep}/{IZO_TEKRAR_SAYISI}): Elini basinin arkasina koy"
        elif izo_ekst_state == "sayimda":
            gecen_saniye = time.time() - izo_ekst_timer; aci_farki = abs(fleksiyon_aci - izo_ekst_aci); bas_dik = (aci_farki < 15) 
            if not el_pozisyonda: izo_ekst_state = "beklemede"; return f"HATA ({izo_ekst_rep}/{IZO_TEKRAR_SAYISI}): Elini indirdin!"
            if not bas_dik: izo_ekst_state = "beklemede"; return f"HILE ({izo_ekst_rep}/{IZO_TEKRAR_SAYISI}): Kafani oynattin!"
            if gecen_saniye >= IZO_SURE_HEDEF:
                izo_ekst_rep += 1; izo_ekst_state = "dinlen"; izo_ekst_timer = time.time(); return f"HARIKA! ({izo_ekst_rep}/{IZO_TEKRAR_SAYISI}). Dinlen."
            else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Izometrik ({izo_ekst_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT! {kalan_saniye}s..."
        elif izo_ekst_state == "dinlen":
            if time.time() - izo_ekst_timer >= IZO_DINLENME_SURESI:
                izo_ekst_state = "beklemede"; return f"Hazir ol ({izo_ekst_rep + 1}/{IZO_TEKRAR_SAYISI})... Elini basinin arkasina koy."
            else: return f"HARIKA! ({izo_ekst_rep}/{IZO_TEKRAR_SAYISI}). Dinlen."
    except Exception as e: return f"Izometrik ({izo_ekst_rep}/{IZO_TEKRAR_SAYISI}): Yuzunu goster"

# Hareket 3 (Yana)
def check_boyun_izometrik_lateral(landmarks):
    global izo_lat_sag_state, izo_lat_sag_timer, izo_lat_sag_rep, izo_lat_sag_aci
    global izo_lat_sol_state, izo_lat_sol_timer, izo_lat_sol_rep, izo_lat_sol_aci
    
    sag_el_pozisyonda = False
    sol_el_pozisyonda = False
    try:
        sag_parmak = get_landmark_coords(landmarks, mp_pose.RIGHT_INDEX); sag_kulak_nokta = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
        mesafe_sag = calculate_distance_3d(sag_parmak, sag_kulak_nokta)
        if mesafe_sag < 0.55: sag_el_pozisyonda = True
    except Exception as e: sag_el_pozisyonda = False
    
    try:
        sol_parmak = get_landmark_coords(landmarks, mp_pose.LEFT_INDEX); sol_kulak_nokta = get_landmark_coords(landmarks, mp_pose.LEFT_EAR)
        mesafe_sol = calculate_distance_3d(sol_parmak, sol_kulak_nokta)
        if mesafe_sol < 0.55: sol_el_pozisyonda = True
    except Exception as e: sol_el_pozisyonda = False

    try:
        sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
        bas_ortasi = get_midpoint(sol_kulak, sag_kulak); sol_omuz = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER)
        sag_omuz = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER); omuz_ortasi = get_midpoint(sol_omuz, sag_omuz)
        p_horizontal = np.array(omuz_ortasi) + np.array([0.1, 0, 0])
        lateral_aci = int(90 - calculate_angle_3d(bas_ortasi, omuz_ortasi, p_horizontal))
        
        # STATE MACHINE...
        if sag_el_pozisyonda and not sol_el_pozisyonda:
            if izo_lat_sag_rep >= IZO_TEKRAR_SAYISI:
                if izo_lat_sag_state != "tamamlandi": log_exercise("Izometrik Lateral", IZO_TEKRAR_SAYISI, "Sag"); izo_lat_sag_state = "tamamlandi"
                return f"Izometrik (Sag): TAMAMLANDI!"
            if izo_lat_sag_state == "beklemede":
                izo_lat_sag_state = "sayimda"; izo_lat_sag_timer = time.time(); izo_lat_sag_aci = lateral_aci 
                return f"Izometrik (Sag) ({izo_lat_sag_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT!"
            elif izo_lat_sag_state == "sayimda":
                gecen_saniye = time.time() - izo_lat_sag_timer; aci_farki = abs(lateral_aci - izo_lat_sag_aci); bas_dik = (aci_farki < 10) 
                if not sag_el_pozisyonda: izo_lat_sag_state = "beklemede"; return f"HATA (Sag): Elini indirdin!"
                if not bas_dik: izo_lat_sag_state = "beklemede"; return f"HILE (Sag): Kafani yana egme!"
                if gecen_saniye >= IZO_SURE_HEDEF:
                    izo_lat_sag_rep += 1; izo_lat_sag_state = "dinlen"; izo_lat_sag_timer = time.time(); return f"HARIKA! (Sag) ({izo_lat_sag_rep}/{IZO_TEKRAR_SAYISI})."
                else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Izometrik (Sag) ({izo_lat_sag_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT! {kalan_saniye}s..."
            elif izo_lat_sag_state == "dinlen":
                if time.time() - izo_lat_sag_timer >= IZO_DINLENME_SURESI:
                    izo_lat_sag_state = "beklemede"; return f"Hazir ol (Sag)... Elini koy."
                else: return f"HARIKA! (Sag) ({izo_lat_sag_rep}/{IZO_TEKRAR_SAYISI}). Dinlen."
        elif sol_el_pozisyonda and not sag_el_pozisyonda:
            if izo_lat_sol_rep >= IZO_TEKRAR_SAYISI:
                if izo_lat_sol_state != "tamamlandi": log_exercise("Izometrik Lateral", IZO_TEKRAR_SAYISI, "Sol"); izo_lat_sol_state = "tamamlandi"
                return f"Izometrik (Sol): TAMAMLANDI!"
            if izo_lat_sol_state == "beklemede":
                izo_lat_sol_state = "sayimda"; izo_lat_sol_timer = time.time(); izo_lat_sol_aci = lateral_aci 
                return f"Izometrik (Sol) ({izo_lat_sol_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT!"
            elif izo_lat_sol_state == "sayimda":
                gecen_saniye = time.time() - izo_lat_sol_timer; aci_farki = abs(lateral_aci - izo_lat_sol_aci); bas_dik = (aci_farki < 10) 
                if not sol_el_pozisyonda: izo_lat_sol_state = "beklemede"; return f"HATA (Sol): Elini indirdin!"
                if not bas_dik: izo_lat_sol_state = "beklemede"; return f"HILE (Sol): Kafani yana egme!"
                if gecen_saniye >= IZO_SURE_HEDEF:
                    izo_lat_sol_rep += 1; izo_lat_sol_state = "dinlen"; izo_lat_sol_timer = time.time(); return f"HARIKA! (Sol) ({izo_lat_sol_rep}/{IZO_TEKRAR_SAYISI})."
                else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Izometrik (Sol) ({izo_lat_sol_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT! {kalan_saniye}s..."
            elif izo_lat_sol_state == "dinlen":
                if time.time() - izo_lat_sol_timer >= IZO_DINLENME_SURESI:
                    izo_lat_sol_state = "beklemede"; return f"Hazir ol (Sol)... Elini koy."
                else: return f"HARIKA! (Sol) ({izo_lat_sol_rep}/{IZO_TEKRAR_SAYISI}). Dinlen."
        else:
            return f"Izometrik ({izo_lat_sag_rep}/{IZO_TEKRAR_SAYISI} Sag | {izo_lat_sol_rep}/{IZO_TEKRAR_SAYISI} Sol) Elini sakagina koy"
    except Exception as e: return f"Izometrik (Yan): Yuzunu goster"

# Hareket 4 (İzo Rotasyon)
def check_boyun_izometrik_rotasyon(landmarks):
    global izo_rot_sag_state, izo_rot_sag_timer, izo_rot_sag_rep, izo_rot_sag_aci
    global izo_rot_sol_state, izo_rot_sol_timer, izo_rot_sol_rep, izo_rot_sol_aci
    
    sag_el_pozisyonda = False
    sol_el_pozisyonda = False
    try:
        sag_parmak = get_landmark_coords(landmarks, mp_pose.RIGHT_INDEX); sag_kulak = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
        mesafe_sag = calculate_distance_3d(sag_parmak, sag_kulak)
        if mesafe_sag < 0.55: sag_el_pozisyonda = True
    except Exception as e: sag_el_pozisyonda = False
    try:
        sol_parmak = get_landmark_coords(landmarks, mp_pose.LEFT_INDEX); sol_kulak = get_landmark_coords(landmarks, mp_pose.LEFT_EAR)
        mesafe_sol = calculate_distance_3d(sol_parmak, sol_kulak)
        if mesafe_sol < 0.55: sol_el_pozisyonda = True
    except Exception as e: sol_el_pozisyonda = False

    try:
        sol_kulak_aci = get_landmark_coords(landmarks, mp_pose.LEFT_EAR); sag_kulak_aci = get_landmark_coords(landmarks, mp_pose.RIGHT_EAR)
        burun = get_landmark_coords(landmarks, mp_pose.NOSE); head_width = sag_kulak_aci[0] - sol_kulak_aci[0]
        if head_width < 0.005: raise Exception("Yuz algilanmadi")
        nose_normalized_pos = (burun[0] - sol_kulak_aci[0]) / head_width
        rotasyon_aci = int(((nose_normalized_pos - 0.5) * 2) * 80)
        
        # STATE MACHINE...
        if sag_el_pozisyonda and not sol_el_pozisyonda:
            if izo_rot_sol_rep >= IZO_TEKRAR_SAYISI: 
                if izo_rot_sol_state != "tamamlandi": log_exercise("Izometrik Rotasyon", IZO_TEKRAR_SAYISI, "Sol"); izo_rot_sol_state = "tamamlandi"
                return f"Izometrik (Sol Donus): TAMAMLANDI!"
            if izo_rot_sol_state == "beklemede":
                izo_rot_sol_state = "sayimda"; izo_rot_sol_timer = time.time(); izo_rot_sol_aci = rotasyon_aci 
                return f"Izometrik (Sol Donus) ({izo_rot_sol_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT!"
            elif izo_rot_sol_state == "sayimda":
                gecen_saniye = time.time() - izo_rot_sol_timer; aci_farki = abs(rotasyon_aci - izo_rot_sol_aci); bas_dondu = (aci_farki > 15) 
                if not sag_el_pozisyonda: izo_rot_sol_state = "beklemede"; return f"HATA (Sol Donus): Elini indirdin!"
                if bas_dondu: izo_rot_sol_state = "beklemede"; return f"HILE (Sol Donus): Kafani dondurme!"
                if gecen_saniye >= IZO_SURE_HEDEF:
                    izo_rot_sol_rep += 1; izo_rot_sol_state = "dinlen"; izo_rot_sol_timer = time.time(); return f"HARIKA! (Sol Donus) ({izo_rot_sol_rep}/{IZO_TEKRAR_SAYISI})."
                else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Izometrik (Sol Donus) ({izo_rot_sol_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT! {kalan_saniye}s..."
            elif izo_rot_sol_state == "dinlen":
                if time.time() - izo_rot_sol_timer >= IZO_DINLENME_SURESI:
                    izo_rot_sol_state = "beklemede"; return f"Hazir ol (Sol Donus)... Elini koy."
                else: return f"HARIKA! (Sol Donus) ({izo_rot_sol_rep}/{IZO_TEKRAR_SAYISI}). Dinlen."
        
        elif sol_el_pozisyonda and not sag_el_pozisyonda:
            if izo_rot_sag_rep >= IZO_TEKRAR_SAYISI: 
                if izo_rot_sag_state != "tamamlandi": log_exercise("Izometrik Rotasyon", IZO_TEKRAR_SAYISI, "Sag"); izo_rot_sag_state = "tamamlandi"
                return f"Izometrik (Sag Donus): TAMAMLANDI!"
            if izo_rot_sag_state == "beklemede":
                izo_rot_sag_state = "sayimda"; izo_rot_sag_timer = time.time(); izo_rot_sag_aci = rotasyon_aci 
                return f"Izometrik (Sag Donus) ({izo_rot_sag_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT!"
            elif izo_rot_sag_state == "sayimda":
                gecen_saniye = time.time() - izo_rot_sag_timer; aci_farki = abs(rotasyon_aci - izo_rot_sag_aci); bas_dondu = (aci_farki > 15)
                if not sol_el_pozisyonda: izo_rot_sag_state = "beklemede"; return f"HATA (Sag Donus): Elini indirdin!"
                if bas_dondu: izo_rot_sag_state = "beklemede"; return f"HILE (Sag Donus): Kafani dondurme!"
                if gecen_saniye >= IZO_SURE_HEDEF:
                    izo_rot_sag_rep += 1; izo_rot_sag_state = "dinlen"; izo_rot_sag_timer = time.time(); return f"HARIKA! (Sag Donus) ({izo_rot_sag_rep}/{IZO_TEKRAR_SAYISI})."
                else: kalan_saniye = IZO_SURE_HEDEF - int(gecen_saniye); return f"Izometrik (Sag Donus) ({izo_rot_sag_rep + 1}/{IZO_TEKRAR_SAYISI}): TUT! {kalan_saniye}s..."
            elif izo_rot_sag_state == "dinlen":
                if time.time() - izo_rot_sag_timer >= IZO_DINLENME_SURESI:
                    izo_rot_sag_state = "beklemede"; return f"Hazir ol (Sag Donus)... Elini koy."
                else: return f"HARIKA! (Sag Donus) ({izo_rot_sag_rep}/{IZO_TEKRAR_SAYISI}). Dinlen."
        
        else:
            return f"Izometrik ({izo_rot_sag_rep}/{IZO_TEKRAR_SAYISI} Sag | {izo_rot_sol_rep}/{IZO_TEKRAR_SAYISI} Sol) Elini dondurmek icin koy"

    except Exception as e: return f"Izometrik (Dondurme): Yuzunu goster"