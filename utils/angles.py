# utils/angles.py
# GÜNCEL TAM KOD (Temiz, 2 fonksiyonlu)

import numpy as np

def calculate_angle_3d(a, b, c):
    """
    Üç adet 3D nokta (landmark) arasındaki açıyı hesaplar.
    """
    
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    ba = a - b
    bc = c - b
    
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0 

    cosine_theta = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    angle_rad = np.arccos(cosine_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_distance_3d(p1, p2):
    """
    İki adet 3D nokta (landmark) arasındaki Öklid mesafesini hesaplar.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    distance = np.linalg.norm(p1 - p2)
    return distance