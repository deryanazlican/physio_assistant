# utils/reports.py
# RAPORLAMA MODÜLÜ - Dosyaları 'raporlar' klasörüne düzenli kaydeder.

import os
from datetime import datetime

def kaydet(hasta_ismi, veriler):
    """
    Raporu proje klasöründeki 'raporlar' klasörüne kaydeder.
    """
    if not veriler:
        return None

    # 1. Klasör yoksa oluştur
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Proje ana dizini
    save_dir = os.path.join(base_dir, "raporlar")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Dosya ismini hazırla (Tarih-Saat)
    zaman_damgasi = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dosya_adi = f"Rapor_{hasta_ismi}_{zaman_damgasi}.txt"
    tam_yol = os.path.join(save_dir, dosya_adi)

    # 3. İçeriği hazırla
    icerik = []
    icerik.append(f"HASTA: {hasta_ismi}")
    icerik.append(f"TARIH: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    icerik.append("-" * 30)
    
    # Hataları ve tekrarları temizleyerek yaz
    for veri in veriler:
        icerik.append(veri)
    
    icerik.append("-" * 30)
    icerik.append("FIZYO ASISTAN AI RAPORU SONU")

    # 4. Dosyayı yaz
    try:
        with open(tam_yol, "w", encoding="utf-8") as f:
            f.write("\n".join(icerik))
        print(f"✅ Rapor şuraya kaydedildi: {tam_yol}")
        return tam_yol
    except Exception as e:
        print(f"❌ Rapor kaydetme hatası: {e}")
        return None