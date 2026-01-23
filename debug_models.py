import requests
import urllib3

# Güvenlik uyarılarını sustur (Kırmızı yazılar çıkmasın)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "AIzaSyBQUzyGcWm9voPr9vvStpoiW37xBykZka0"
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

print("📡 Google'a bağlanılıyor (SSL Koruması Devre Dışı)...")

try:
    # verify=False diyerek SSL sertifika kontrolünü kapatıyoruz
    response = requests.get(url, verify=False, timeout=10)

    if response.status_code == 200:
        data = response.json()
        print("\n✅ BAŞARILI! Engel aşıldı. Çalışan modeller:")
        print("-" * 50)
        models = data.get('models', [])
        for m in models:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                # Model isminin başındaki 'models/' kısmını temizleyip yazdıralım
                temiz_isim = m['name'].replace("models/", "")
                print(f"👉 {temiz_isim}")
        print("-" * 50)
        print("Lütfen bu listeden bir isim seç (örn: gemini-1.5-flash) ve bana söyle.")
    else:
        print(f"\n❌ HATA! Kod: {response.status_code}")
        print("Mesaj:", response.text)

except Exception as e:
    print(f"\n❌ Bağlantı Hatası: {e}")