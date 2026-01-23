# utils/ai_coach.py
# DETAYLI DOKTOR ANALİZİ

import requests
import json
import urllib3

# Güvenlik uyarılarını kapat
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
API_KEY = "AIzaSyBQUzyGcWm9voPr9vvStpoiW37xBykZka0"

def doktor_yorumu_al(egzersiz_adi, tekrar_sayisi, hatalar_listesi):
    try:
        print("⏳ AI Doktor Düşünüyor... (Detaylı Analiz Modu)")

        # Hataları temizle
        if not hatalar_listesi:
            durum_metni = "Hasta hareketi kusursuz, hatasız bir formda tamamladı."
        else:
            hatalar_temiz = ", ".join(list(set(hatalar_listesi)))
            durum_metni = f"Tespit edilen problemler: {hatalar_temiz}"

        # --- YENİ PROMPT: DETAYLI VE KONUŞKAN ---
        prompt = f"""
        Sen tecrübeli ve ilgili bir Fizyoterapistsin.
        Hastanın egzersiz verilerini inceleyip ona DETAYLI bir geri bildirim ver.
        
        VERİLER:
        - Egzersiz: {egzersiz_adi}
        - Yapılan Tekrar: {tekrar_sayisi}
        - Performans Durumu: {durum_metni}
        
        KURALLAR:
        1. Asla tek cümle kurma. En az 3-4 cümlelik, doyurucu bir paragraf yaz.
        2. Eğer hata yaptıysa; hatanın neden zararlı olduğunu ve doğrusunu nasıl yapacağını anlat.
        3. Eğer hatasız yaptıysa; hangi kasların çalıştığını ve neden iyi yaptığını söyleyerek motive et.
        4. Tıbbi terimleri halk diliyle açıkla.
        5. Samimi ol ama ciddiyetini koru.
        """

        # Model listesi (Sırayla dener)
        models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": prompt}]}]}

        for model in models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
                response = requests.post(url, headers=headers, json=data, verify=False, timeout=8)
                if response.status_code == 200:
                    text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                    # Çok uzunsa ekrana sığması için kırpma yapılabilir ama şimdilik ham hali gelsin
                    print(f"✅ AI Yorumu Geldi ({len(text)} karakter)")
                    return text
            except: continue

        return "Şu an sunuculara ulaşılamıyor. Ancak genel olarak hareket formunuza dikkat etmenizi öneririm."

    except Exception as e:
        print(f"Hata: {e}")
        return "Bağlantı sorunu oluştu."