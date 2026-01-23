# core/plan_generator.py
import json
import os
from datetime import datetime, timedelta

class PersonalizedPlanGenerator:
    """
    Hastaya özel egzersiz planı oluşturur
    """
    
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        os.makedirs(data_folder, exist_ok=True)
        
        # Egzersiz şablonları
        self.exercise_templates = {
            "BOYUN_AGRISI": {
                "Hafta_1": {
                    "Pazartesi": ["ROM_LAT", "ROM_ROT"],
                    "Çarşamba": ["ROM_FLEKS", "IZO_FLEKS"],
                    "Cuma": ["ROM_CEMBER", "IZO_LAT"]
                },
                "Hafta_2": {
                    "Pazartesi": ["ROM_LAT", "ROM_ROT", "IZO_FLEKS"],
                    "Çarşamba": ["ROM_FLEKS", "IZO_EKST", "ROM_CEMBER"],
                    "Cuma": ["ROM_LAT", "ROM_ROT", "IZO_LAT"]
                }
            },
            "OMUZ_AGRISI": {
                "Hafta_1": {
                    "Pazartesi": ["OMUZ_PEN_FLEKSIYON", "OMUZ_PEN_ABDUKSIYON"],
                    "Çarşamba": ["OMUZ_YANA_ACMA", "OMUZ_GERME"],
                    "Cuma": ["OMUZ_CEMBER", "OMUZ_DUVAR_YANA"]
                },
                "Hafta_2": {
                    "Pazartesi": ["OMUZ_ONE_ACMA", "OMUZ_DISA_ACMA"],
                    "Çarşamba": ["OMUZ_DUVAR_ONE", "OMUZ_DUVAR_GERIYE"],
                    "Cuma": ["OMUZ_YANA_ACMA", "OMUZ_CEMBER"]
                }
            },
            "DIZ_AGRISI": {
                "Hafta_1": {
                    "Pazartesi": ["DIZ_HAVLU_EZME", "DIZ_OTUR_UZAT"],
                    "Çarşamba": ["DIZ_YUZUSTU_BUKME", "DIZ_YAN_KALDIR"],
                    "Cuma": ["DIZ_DUVAR_SQUAT", "DIZ_HAVLU_EZME"]
                }
            },
            "KALCA_AGRISI": {
                "Hafta_1": {
                    "Pazartesi": ["KALCA_DIZ_CEKME", "KALCA_KOPRU"],
                    "Çarşamba": ["KALCA_DUZ_KALDIR", "KALCA_YAN_ACMA"],
                    "Cuma": ["KALCA_YUZUSTU", "KALCA_YAN_DIZ_CEKME"]
                }
            },
            "BEL_AGRISI": {
                "Hafta_1": {
                    "Pazartesi": ["BEL_TEK_DIZ", "BEL_KOPRU"],
                    "Çarşamba": ["BEL_CIFT_DIZ", "BEL_KEDI_DEVE"],
                    "Cuma": ["BEL_MEKIK", "BEL_SLR"]
                }
            }
        }
    
    def create_plan(self, patient_name, condition, fitness_level=5, weeks=2):
        """
        Kişiselleştirilmiş plan oluştur
        
        Args:
            patient_name: Hasta adı
            condition: Durum (BOYUN_AGRISI, OMUZ_AGRISI vb.)
            fitness_level: 1-10 arası kondisyon seviyesi
            weeks: Kaç haftalık plan
        
        Returns:
            dict: Plan detayları
        """
        
        if condition not in self.exercise_templates:
            condition = "BOYUN_AGRISI"  # Varsayılan
        
        template = self.exercise_templates[condition]
        
        # Plan oluştur
        plan = {
            "patient_name": patient_name,
            "condition": condition,
            "fitness_level": fitness_level,
            "created_date": datetime.now().isoformat(),
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "weeks": weeks,
            "schedule": {}
        }
        
        # Haftaları doldur
        current_date = datetime.now()
        for week in range(1, weeks + 1):
            week_key = f"Hafta_{week}"
            
            if week_key in template:
                week_template = template[week_key]
            else:
                # Son haftayı tekrarla
                week_template = template[f"Hafta_{len(template)}"]
            
            plan["schedule"][week_key] = {}
            
            for day, exercises in week_template.items():
                # Fitness seviyesine göre ayarla
                if fitness_level < 3:
                    # Düşük kondisyon: İlk egzersizi al
                    adjusted_exercises = exercises[:1]
                elif fitness_level < 7:
                    # Orta kondisyon: İlk 2 egzersiz
                    adjusted_exercises = exercises[:2]
                else:
                    # Yüksek kondisyon: Hepsini al
                    adjusted_exercises = exercises
                
                plan["schedule"][week_key][day] = {
                    "exercises": adjusted_exercises,
                    "completed": False,
                    "date": None
                }
        
        # Kaydet
        self._save_plan(patient_name, plan)
        
        return plan
    
    def _save_plan(self, patient_name, plan):
        """Planı JSON'a kaydet"""
        filename = f"{self.data_folder}/{patient_name}_plan.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"✅ Plan kaydedildi: {filename}")
    
    def load_plan(self, patient_name):
        """Planı yükle"""
        filename = f"{self.data_folder}/{patient_name}_plan.json"
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_today_exercises(self, patient_name):
        """Bugünkü egzersizleri getir"""
        plan = self.load_plan(patient_name)
        if not plan:
            return None
        
        today = datetime.now()
        start_date = datetime.fromisoformat(plan["created_date"])
        days_passed = (today - start_date).days
        
        # Hangi hafta?
        current_week = (days_passed // 7) + 1
        if current_week > plan["weeks"]:
            return {"message": "Plan tamamlandı!", "exercises": []}
        
        # Hangi gün?
        day_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        today_name = day_names[today.weekday()]
        
        week_key = f"Hafta_{current_week}"
        if week_key in plan["schedule"] and today_name in plan["schedule"][week_key]:
            day_plan = plan["schedule"][week_key][today_name]
            return {
                "week": current_week,
                "day": today_name,
                "exercises": day_plan["exercises"],
                "completed": day_plan["completed"]
            }
        
        return {"message": "Bugün dinlenme günü", "exercises": []}
    
    def mark_day_complete(self, patient_name, week, day):
        """Günü tamamlandı olarak işaretle"""
        plan = self.load_plan(patient_name)
        if plan:
            week_key = f"Hafta_{week}"
            if week_key in plan["schedule"] and day in plan["schedule"][week_key]:
                plan["schedule"][week_key][day]["completed"] = True
                plan["schedule"][week_key][day]["date"] = datetime.now().isoformat()
                self._save_plan(patient_name, plan)
                return True
        return False
    
    def get_progress_summary(self, patient_name):
        """İlerleme özeti"""
        plan = self.load_plan(patient_name)
        if not plan:
            return None
        
        total_days = 0
        completed_days = 0
        
        for week_key, week_data in plan["schedule"].items():
            for day, day_data in week_data.items():
                total_days += 1
                if day_data["completed"]:
                    completed_days += 1
        
        completion_rate = (completed_days / total_days * 100) if total_days > 0 else 0
        
        return {
            "total_days": total_days,
            "completed_days": completed_days,
            "completion_rate": completion_rate,
            "current_week": self._get_current_week(plan)
        }
    
    def _get_current_week(self, plan):
        """Mevcut haftayı hesapla"""
        start_date = datetime.fromisoformat(plan["created_date"])
        today = datetime.now()
        days_passed = (today - start_date).days
        return min((days_passed // 7) + 1, plan["weeks"])


# ==================== KULLANIM ÖRNEĞİ ====================
if __name__ == "__main__":
    planner = PersonalizedPlanGenerator()
    
    # 1. Plan oluştur
    print("1. Plan oluşturuluyor...")
    plan = planner.create_plan(
        patient_name="DERYA",
        condition="BOYUN_AGRISI",
        fitness_level=6,
        weeks=2
    )
    
    print(f"Plan oluşturuldu: {plan['patient_name']}")
    print(f"Durum: {plan['condition']}")
    print(f"Hafta sayısı: {plan['weeks']}")
    
    # 2. Bugünkü egzersizleri getir
    print("\n2. Bugünkü egzersizler:")
    today = planner.get_today_exercises("DERYA")
    if today:
        print(f"Hafta: {today.get('week', 'N/A')}")
        print(f"Gün: {today.get('day', 'N/A')}")
        print(f"Egzersizler: {today.get('exercises', [])}")
    
    # 3. İlerleme özeti
    print("\n3. İlerleme özeti:")
    summary = planner.get_progress_summary("DERYA")
    if summary:
        print(f"Tamamlanan günler: {summary['completed_days']}/{summary['total_days']}")
        print(f"Tamamlanma oranı: %{summary['completion_rate']:.1f}")
        print(f"Mevcut hafta: {summary['current_week']}")