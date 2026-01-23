# ai/pain_predictor.py
import json
import os
import numpy as np

class SimplePainPredictor:
    """
    Basit kural tabanlı ağrı tahmin modeli
    (Gerçek ML modeli için sklearn gerekir)
    """
    
    def __init__(self):
        # Kural tabanlı tahmin parametreleri
        self.rules = {
            'overwork': {  # Aşırı çalışma
                'reps_threshold': 15,
                'duration_threshold': 20,  # dakika
                'pain_increase': 2
            },
            'poor_form': {  # Kötü form
                'quality_threshold': 0.5,
                'pain_increase': 1.5
            },
            'insufficient_rest': {  # Yetersiz dinlenme
                'min_rest_hours': 24,
                'pain_increase': 1
            },
            'progression_too_fast': {  # Çok hızlı ilerleme
                'angle_increase_threshold': 20,  # Haftalık %20 artış
                'pain_increase': 1.5
            }
        }
    
    def predict_pain_after_exercise(self, current_data, exercise_history=None):
        """
        Egzersiz sonrası ağrı tahmini
        
        Args:
            current_data: {
                'exercise': str,
                'reps': int,
                'duration': float (dk),
                'quality': float (0-1),
                'current_pain': int (0-10),
                'last_exercise_hours_ago': float
            }
            exercise_history: list (opsiyonel)
        
        Returns:
            dict: {
                'predicted_pain': int (0-10),
                'risk_level': str (Düşük/Orta/Yüksek),
                'warnings': list,
                'recommendations': list
            }
        """
        
        # Başlangıç ağrısı
        base_pain = current_data.get('current_pain', 3)
        predicted_pain = base_pain
        warnings = []
        recommendations = []
        
        # Kural 1: Aşırı çalışma kontrolü
        if current_data.get('reps', 0) > self.rules['overwork']['reps_threshold']:
            predicted_pain += self.rules['overwork']['pain_increase']
            warnings.append("⚠️ Çok fazla tekrar yapıyorsunuz")
            recommendations.append("Tekrar sayısını azaltın (10-12 yeterli)")
        
        if current_data.get('duration', 0) > self.rules['overwork']['duration_threshold']:
            predicted_pain += self.rules['overwork']['pain_increase']
            warnings.append("⚠️ Egzersiz süresi çok uzun")
            recommendations.append("Toplam süreyi 15 dakikada tutun")
        
        # Kural 2: Form kalitesi
        quality = current_data.get('quality', 1.0)
        if quality < self.rules['poor_form']['quality_threshold']:
            predicted_pain += self.rules['poor_form']['pain_increase']
            warnings.append("⚠️ Form kalitesi düşük")
            recommendations.append("Yavaşlayın ve formunuza odaklanın")
        
        # Kural 3: Dinlenme süresi
        last_exercise = current_data.get('last_exercise_hours_ago', 48)
        if last_exercise < self.rules['insufficient_rest']['min_rest_hours']:
            predicted_pain += self.rules['insufficient_rest']['pain_increase']
            warnings.append("⚠️ Yeterince dinlenmediniz")
            recommendations.append("Egzersizler arası en az 24 saat bekleyin")
        
        # Kural 4: İlerleme kontrolü (eğer geçmiş varsa)
        if exercise_history and len(exercise_history) >= 2:
            recent_angles = [ex['data'].get('angle', 0) for ex in exercise_history[-5:]]
            if recent_angles:
                angle_increase = (recent_angles[-1] - recent_angles[0]) / len(recent_angles)
                if angle_increase > 5:  # Haftada 5° artış çok
                    predicted_pain += self.rules['progression_too_fast']['pain_increase']
                    warnings.append("⚠️ Çok hızlı ilerliyorsunuz")
                    recommendations.append("İlerlemeyi yavaşlatın (%10 kural)")
        
        # Pozitif faktörler (ağrıyı azaltan)
        if quality > 0.8:
            predicted_pain -= 0.5
            recommendations.append("✅ Formunuz mükemmel, böyle devam!")
        
        if last_exercise > 48:
            predicted_pain -= 0.5
            recommendations.append("✅ İyi dinlendiniz")
        
        # Sınırla
        predicted_pain = max(0, min(10, predicted_pain))
        
        # Risk seviyesi
        if predicted_pain <= 3:
            risk_level = "Düşük"
            risk_color = "🟢"
        elif predicted_pain <= 6:
            risk_level = "Orta"
            risk_color = "🟡"
        else:
            risk_level = "Yüksek"
            risk_color = "🔴"
        
        return {
            'predicted_pain': round(predicted_pain, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'warnings': warnings,
            'recommendations': recommendations if recommendations else ["✅ Her şey yolunda, devam edin!"]
        }
    
    def should_continue(self, prediction):
        """Egzersize devam edilmeli mi?"""
        return prediction['predicted_pain'] <= 6
    
    def get_recommendation_text(self, prediction):
        """Öneri metnini al"""
        text = f"{prediction['risk_color']} Risk Seviyesi: {prediction['risk_level']}\n"
        text += f"Tahmini Ağrı: {prediction['predicted_pain']}/10\n\n"
        
        if prediction['warnings']:
            text += "⚠️ UYARILAR:\n"
            for warning in prediction['warnings']:
                text += f"  • {warning}\n"
            text += "\n"
        
        text += "💡 ÖNERİLER:\n"
        for rec in prediction['recommendations']:
            text += f"  • {rec}\n"
        
        return text


# ==================== GELİŞMİŞ ML SÜRÜMÜ (Opsiyonel) ====================
try:
    from sklearn.ensemble import RandomForestRegressor
    import pickle
    
    class MLPainPredictor:
        """
        Makine öğrenmesi tabanlı ağrı tahmini
        """
        
        def __init__(self, model_path="pain_model.pkl"):
            self.model_path = model_path
            self.model = None
            self.is_trained = False
            
            # Model varsa yükle
            if os.path.exists(model_path):
                self.load_model()
        
        def train(self, training_data):
            """
            Modeli eğit
            
            training_data: list of {
                'features': [reps, duration, quality, current_pain, rest_hours],
                'label': pain_after (0-10)
            }
            """
            if len(training_data) < 10:
                print("⚠️ Yeterli veri yok (en az 10 kayıt)")
                return False
            
            X = np.array([d['features'] for d in training_data])
            y = np.array([d['label'] for d in training_data])
            
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            self.is_trained = True
            
            self.save_model()
            print(f"✅ Model eğitildi ({len(training_data)} kayıt)")
            return True
        
        def predict(self, features):
            """Tahmin yap"""
            if not self.is_trained:
                return None
            
            prediction = self.model.predict([features])[0]
            return max(0, min(10, prediction))
        
        def save_model(self):
            """Modeli kaydet"""
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
        
        def load_model(self):
            """Modeli yükle"""
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                print("✅ Model yüklendi")
            except:
                print("⚠️ Model yüklenemedi")

except ImportError:
    print("ℹ️ sklearn yüklü değil, sadece basit tahmin kullanılabilir")


# ==================== KULLANIM ÖRNEĞİ ====================
if __name__ == "__main__":
    predictor = SimplePainPredictor()
    
    # Test senaryoları
    print("TEST 1: Normal Egzersiz")
    print("="*50)
    prediction = predictor.predict_pain_after_exercise({
        'exercise': 'ROM_LAT',
        'reps': 10,
        'duration': 10,
        'quality': 0.8,
        'current_pain': 3,
        'last_exercise_hours_ago': 48
    })
    print(predictor.get_recommendation_text(prediction))
    
    print("\n\nTEST 2: Aşırı Çalışma")
    print("="*50)
    prediction = predictor.predict_pain_after_exercise({
        'exercise': 'ROM_LAT',
        'reps': 20,  # Çok fazla!
        'duration': 25,  # Çok uzun!
        'quality': 0.6,  # Kötü form
        'current_pain': 5,
        'last_exercise_hours_ago': 12  # Az dinlenme
    })
    print(predictor.get_recommendation_text(prediction))
    
    print(f"\nDevam edilmeli mi? {predictor.should_continue(prediction)}")