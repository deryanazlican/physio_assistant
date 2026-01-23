# core/analytics.py
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict
import numpy as np

class ProgressAnalytics:
    """
    Hasta ilerlemesini takip eder ve görselleştirir
    """
    
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        os.makedirs(data_folder, exist_ok=True)
        
        # Matplotlib Türkçe desteği
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def record_exercise(self, patient_name, exercise_name, data):
        """
        Egzersiz kaydı oluştur
        
        Args:
            patient_name: Hasta adı
            exercise_name: Egzersiz kodu
            data: {
                'reps': int,
                'angle': float,
                'duration': float,
                'quality': float (0-1),
                'pain_level': int (0-10),
                'notes': str
            }
        """
        filename = f"{self.data_folder}/{patient_name}_history.json"
        
        # Mevcut geçmişi yükle
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # Yeni kayıt ekle
        record = {
            'timestamp': datetime.now().isoformat(),
            'exercise': exercise_name,
            'data': data
        }
        history.append(record)
        
        # Kaydet
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def get_exercise_history(self, patient_name, exercise_name=None, days=30):
        """Egzersiz geçmişini getir"""
        filename = f"{self.data_folder}/{patient_name}_history.json"
        
        if not os.path.exists(filename):
            return []
        
        with open(filename, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # Filtreleme
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered = []
        
        for record in history:
            record_date = datetime.fromisoformat(record['timestamp'])
            if record_date >= cutoff_date:
                if exercise_name is None or record['exercise'] == exercise_name:
                    filtered.append(record)
        
        return filtered
    
    def plot_rom_progress(self, patient_name, exercise_name, save_path="progress_rom.png"):
        """ROM (Hareket Açıklığı) gelişimi grafiği"""
        history = self.get_exercise_history(patient_name, exercise_name, days=30)
        
        if len(history) < 2:
            print("Yeterli veri yok (en az 2 kayıt gerekli)")
            return None
        
        # Verileri hazırla
        dates = []
        angles = []
        
        for record in history:
            date = datetime.fromisoformat(record['timestamp'])
            angle = record['data'].get('angle', 0)
            
            dates.append(date)
            angles.append(angle)
        
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, angles, marker='o', linestyle='-', linewidth=2, markersize=8, color='#00CED1')
        
        # Trend çizgisi (lineer regresyon)
        if len(dates) > 2:
            x_numeric = mdates.date2num(dates)
            z = np.polyfit(x_numeric, angles, 1)
            p = np.poly1d(z)
            ax.plot(dates, p(x_numeric), linestyle='--', color='red', alpha=0.7, label='Trend')
        
        # Başlangıç ve son değerleri göster
        ax.text(dates[0], angles[0], f'{angles[0]:.1f}°', ha='right', va='bottom', fontsize=10, color='green')
        ax.text(dates[-1], angles[-1], f'{angles[-1]:.1f}°', ha='left', va='top', fontsize=10, color='green')
        
        # İyileşme oranı
        improvement = angles[-1] - angles[0]
        improvement_pct = (improvement / angles[0] * 100) if angles[0] != 0 else 0
        
        # Başlık ve etiketler
        ax.set_title(f'{exercise_name} - Hareket Açıklığı Gelişimi\nİyileşme: {improvement:.1f}° (%{improvement_pct:.1f})', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Tarih', fontsize=12)
        ax.set_ylabel('Açı (Derece)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Tarih formatı
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafik kaydedildi: {save_path}")
        
        return save_path
    
    def plot_pain_trend(self, patient_name, save_path="progress_pain.png"):
        """Ağrı seviyesi trendi"""
        history = self.get_exercise_history(patient_name, days=30)
        
        if len(history) < 2:
            print("Yeterli veri yok")
            return None
        
        # Günlük ortalama ağrı
        daily_pain = defaultdict(list)
        
        for record in history:
            date = datetime.fromisoformat(record['timestamp']).date()
            pain = record['data'].get('pain_level', 0)
            daily_pain[date].append(pain)
        
        # Ortalama hesapla
        dates = sorted(daily_pain.keys())
        avg_pain = [np.mean(daily_pain[d]) for d in dates]
        
        # Grafik
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, avg_pain, marker='o', linestyle='-', linewidth=2, markersize=8, color='#FF6347')
        ax.fill_between(dates, avg_pain, alpha=0.3, color='#FF6347')
        
        # Renk bölgeleri
        ax.axhspan(0, 3, alpha=0.1, color='green', label='Düşük (0-3)')
        ax.axhspan(3, 7, alpha=0.1, color='orange', label='Orta (3-7)')
        ax.axhspan(7, 10, alpha=0.1, color='red', label='Yüksek (7-10)')
        
        ax.set_title('Ağrı Seviyesi Trendi (Günlük Ortalama)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tarih', fontsize=12)
        ax.set_ylabel('Ağrı Seviyesi (0-10)', fontsize=12)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Ağrı grafiği kaydedildi: {save_path}")
        
        return save_path
    
    def plot_weekly_activity(self, patient_name, save_path="progress_activity.png"):
        """Haftalık aktivite grafiği"""
        history = self.get_exercise_history(patient_name, days=30)
        
        if len(history) < 1:
            print("Veri yok")
            return None
        
        # Haftalık egzersiz sayısı
        weekly_counts = defaultdict(int)
        
        for record in history:
            date = datetime.fromisoformat(record['timestamp'])
            week_start = date - timedelta(days=date.weekday())
            week_key = week_start.strftime('%d/%m')
            weekly_counts[week_key] += 1
        
        weeks = sorted(weekly_counts.keys())
        counts = [weekly_counts[w] for w in weeks]
        
        # Grafik
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(weeks, counts, color='#4169E1', alpha=0.7, edgecolor='black')
        
        # Değerleri göster
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('Haftalık Egzersiz Aktivitesi', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hafta Başlangıcı', fontsize=12)
        ax.set_ylabel('Egzersiz Sayısı', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Aktivite grafiği kaydedildi: {save_path}")
        
        return save_path
    
    def generate_summary_report(self, patient_name):
        """Özet rapor oluştur"""
        history = self.get_exercise_history(patient_name, days=30)
        
        if not history:
            return {"error": "Veri bulunamadı"}
        
        # İstatistikler
        total_exercises = len(history)
        
        # Egzersiz dağılımı
        exercise_counts = defaultdict(int)
        total_reps = 0
        total_duration = 0
        pain_levels = []
        
        for record in history:
            exercise_counts[record['exercise']] += 1
            total_reps += record['data'].get('reps', 0)
            total_duration += record['data'].get('duration', 0)
            pain_levels.append(record['data'].get('pain_level', 0))
        
        avg_pain = np.mean(pain_levels) if pain_levels else 0
        
        # En çok yapılan egzersiz
        most_common = max(exercise_counts.items(), key=lambda x: x[1]) if exercise_counts else ("Yok", 0)
        
        # İlk ve son tarih
        first_date = datetime.fromisoformat(history[0]['timestamp'])
        last_date = datetime.fromisoformat(history[-1]['timestamp'])
        active_days = (last_date - first_date).days + 1
        
        report = {
            "patient_name": patient_name,
            "period": f"{first_date.strftime('%d/%m/%Y')} - {last_date.strftime('%d/%m/%Y')}",
            "active_days": active_days,
            "total_exercises": total_exercises,
            "total_reps": total_reps,
            "total_duration_min": round(total_duration / 60, 1),
            "avg_pain_level": round(avg_pain, 1),
            "most_common_exercise": most_common[0],
            "exercise_distribution": dict(exercise_counts)
        }
        
        return report
    
    def print_report(self, patient_name):
        """Raporu ekrana yazdır"""
        report = self.generate_summary_report(patient_name)
        
        if "error" in report:
            print(report["error"])
            return
        
        print("\n" + "="*50)
        print(f"   İLERLEME RAPORU - {report['patient_name']}")
        print("="*50)
        print(f"📅 Dönem: {report['period']}")
        print(f"📊 Aktif Gün: {report['active_days']}")
        print(f"🏋️ Toplam Egzersiz: {report['total_exercises']}")
        print(f"🔄 Toplam Tekrar: {report['total_reps']}")
        print(f"⏱️ Toplam Süre: {report['total_duration_min']} dk")
        print(f"😊 Ortalama Ağrı: {report['avg_pain_level']}/10")
        print(f"⭐ En Çok Yapılan: {report['most_common_exercise']}")
        print("="*50 + "\n")


# ==================== KULLANIM ÖRNEĞİ ====================
if __name__ == "__main__":
    analytics = ProgressAnalytics()
    
    # Test verisi ekle
    print("Test verisi oluşturuluyor...")
    patient = "DERYA"
    
    for i in range(10):
        analytics.record_exercise(
            patient_name=patient,
            exercise_name="ROM_LAT",
            data={
                'reps': 10,
                'angle': 30 + i * 5,  # Gelişim simülasyonu
                'duration': 120,
                'quality': 0.7 + i * 0.03,
                'pain_level': max(0, 7 - i),  # Ağrı azalıyor
                'notes': f'Gün {i+1}'
            }
        )
    
    print(f"✅ {10} test kaydı eklendi\n")
    
    # Grafikler
    print("Grafikler oluşturuluyor...")
    analytics.plot_rom_progress(patient, "ROM_LAT")
    analytics.plot_pain_trend(patient)
    analytics.plot_weekly_activity(patient)
    
    # Rapor
    analytics.print_report(patient)