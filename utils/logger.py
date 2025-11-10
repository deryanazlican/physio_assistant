# utils/logger.py
# YENİ DOSYA - Egzersiz Raporlama (Logging) Modülü

import csv
import os
from datetime import datetime

# Rapor dosyasının adı
LOG_FILE = 'egzersiz_raporu.csv'
# Raporun başlıkları
FIELDNAMES = ['Tarih', 'Saat', 'Egzersiz Adi', 'Tekrar Sayisi', 'Yon']

def log_exercise(exercise_name, reps, side=None):
    """
    Tamamlanan bir egzersiz setini 'egzersiz_raporu.csv' dosyasına kaydeder.
    """
    
    # Dosya yoksa, başlık satırını ekle
    file_exists = os.path.isfile(LOG_FILE)
    
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            
            if not file_exists:
                writer.writeheader() # Başlıkları yaz
            
            # Veriyi yaz
            writer.writerow({
                'Tarih': datetime.now().strftime('%Y-%m-%d'),
                'Saat': datetime.now().strftime('%H:%M:%S'),
                'Egzersiz Adi': exercise_name,
                'Tekrar Sayisi': reps,
                'Yon': side if side else 'N/A' # Yön yoksa N/A yaz
            })
            
        print(f"RAPORLANDI: {exercise_name} - {reps} tekrar ({side})")
        
    except Exception as e:
        print(f"Raporlama Hatasi: {e}")