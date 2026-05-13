# ai/gemini_vision.py
from __future__ import annotations

import time
import cv2
from google import genai
from google.genai import types


class GeminiVisionAnalyzer:
    """
    Gemini Vision ile egzersiz formu analizi (google-genai).
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-flash"):
        self.enabled = False
        self.client = None
        self.model_name = model_name

        if not api_key:
            print("⚠️ Gemini API key bulunamadı. Vision özelliği devre dışı.")
            return

        try:
            self.client = genai.Client(api_key=api_key)
            self.enabled = True
            print(f"✅ Gemini Vision hazır! (model={self.model_name})")
        except Exception as e:
            print(f"⚠️ Gemini Vision başlatma hatası: {e}")
            self.enabled = False

        self.last_analysis_time = 0
        self.analysis_cooldown = 10  # saniye

    def analyze_exercise_form(self, frame, exercise_name: str, current_angle: float | None = None):
        if not self.enabled or not self.client:
            return None

        now = time.time()
        if now - self.last_analysis_time < self.analysis_cooldown:
            return None

        try:
            # frame (BGR) -> JPEG bytes
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            image_bytes = buf.tobytes()

            prompt = self._create_prompt(exercise_name, current_angle)

            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt
                ],
            )

            self.last_analysis_time = now
            text = (resp.text or "").strip()
            if not text:
                return None

            return self._parse_response(text)

        except Exception as e:
            print(f"Gemini Vision hatası: {e}")
            return None

    def _create_prompt(self, exercise_name: str, current_angle: float | None):
        exercise_instructions = {
            "ROM_LAT": "Yana eğilme. Kulak omuza yaklaşmalı, omuzlar sabit kalmalı.",
            "ROM_ROT": "Boyun rotasyonu. Baş yana dönmeli, omuzlar sabit.",
            "OMUZ_YANA_ACMA": "Kollar yana kaldırılıyor. Omuzlar aşağıda, sırt düz.",
            "DIZ_HAVLU_EZME": "Diz altındaki havlu eziliyor. Bacak düz, topuk yerde.",
            "KALCA_KOPRU": "Köprü. Kalça havada, gövde düz, boyun nötr."
        }
        instruction = exercise_instructions.get(exercise_name, "Fizyoterapi egzersizi")

        prompt = f"""
Sen bir uzman fizyoterapistsin. Bu görüntüde hasta {exercise_name} egzersizini yapıyor.

EGZERSİZ: {instruction}

GÖREVİN:
1) Postür/form analizi
2) Hataları tespit et (kompansasyon/yanlış hizalama)
3) 3 somut öneri ver

CEVAP FORMATI (Türkçe, kısa ve net):
FORM KALİTESİ: [Mükemmel/İyi/Orta/Zayıf]
HATALAR: [Varsa liste, yoksa "Yok"]
ÖNERİLER: [3 madde]

Maksimum 100 kelime.
""".strip()

        if current_angle is not None:
            prompt += f"\n\nMEVCUT AÇI: {current_angle:.1f}°"

        return prompt

    def _parse_response(self, response_text: str):
        quality_map = {
            "Mükemmel": 1.0,
            "İyi": 0.8,
            "Orta": 0.6,
            "Zayıf": 0.4
        }

        quality_score = 0.7
        for word, score in quality_map.items():
            if word in response_text:
                quality_score = score
                break

        suggestions = []
        if "ÖNERİLER:" in response_text:
            part = response_text.split("ÖNERİLER:", 1)[1].strip()
            suggestions = [s.strip("•- \t") for s in part.splitlines() if s.strip()]

        return {
            "feedback": response_text,
            "quality_score": quality_score,
            "suggestions": suggestions[:3]
        }