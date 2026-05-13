# core/plan_generator.py
import json
import os
from datetime import datetime
import re
from pathlib import Path


class PersonalizedPlanGenerator:
    """
    Hastaya özel egzersiz planı oluşturur
    """

    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        os.makedirs(data_folder, exist_ok=True)

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

    def _safe_patient_id(self, patient_name: str) -> str:
        name = (patient_name or "UNKNOWN").strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
        return name or "UNKNOWN"

    def _plan_path(self, patient_name: str) -> str:
        pid = self._safe_patient_id(patient_name)
        return str(Path(self.data_folder) / f"{pid}_plan.json")

    def _atomic_save(self, filename: str, plan: dict):
        tmp = filename + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        os.replace(tmp, filename)

    def _normalize_condition(self, condition):
        if condition is None:
            return None

        condition = str(condition).strip().upper()
        condition = condition.replace("İ", "I")
        condition = condition.replace("Ç", "C")
        condition = condition.replace("Ğ", "G")
        condition = condition.replace("Ö", "O")
        condition = condition.replace("Ş", "S")
        condition = condition.replace("Ü", "U")

        aliases = {
            "BOYUN_AGRISI": "BOYUN_AGRISI",
            "BOYUN AĞRISI": "BOYUN_AGRISI",
            "BOYUN AGRISI": "BOYUN_AGRISI",

            "OMUZ_AGRISI": "OMUZ_AGRISI",
            "OMUZ AĞRISI": "OMUZ_AGRISI",
            "OMUZ AGRISI": "OMUZ_AGRISI",

            "DIZ_AGRISI": "DIZ_AGRISI",
            "DIZ AĞRISI": "DIZ_AGRISI",
            "DIZ AGRISI": "DIZ_AGRISI",

            "KALCA_AGRISI": "KALCA_AGRISI",
            "KALCA AĞRISI": "KALCA_AGRISI",
            "KALCA AGRISI": "KALCA_AGRISI",

            "BEL_AGRISI": "BEL_AGRISI",
            "BEL AĞRISI": "BEL_AGRISI",
            "BEL AGRISI": "BEL_AGRISI",
        }

        return aliases.get(condition, condition)

    def create_plan(self, patient_name, condition, fitness_level=5, weeks=2):
        condition_raw = condition
        condition = self._normalize_condition(condition)

        print(f"[PLAN_GENERATOR] raw condition = {condition_raw!r}")
        print(f"[PLAN_GENERATOR] normalized condition = {condition!r}")
        print(f"[PLAN_GENERATOR] available keys = {list(self.exercise_templates.keys())}")

        if condition not in self.exercise_templates:
            raise ValueError(
                f"Geçersiz condition geldi: {condition_raw!r} -> {condition!r}"
            )

        template = self.exercise_templates[condition]

        plan = {
            "patient_name": patient_name,
            "condition": condition,
            "fitness_level": fitness_level,
            "created_date": datetime.now().isoformat(),
            "start_date": datetime.now().date().isoformat(),
            "weeks": weeks,
            "schedule": {}
        }

        for week in range(1, weeks + 1):
            week_key = f"Hafta_{week}"

            if week_key in template:
                week_template = template[week_key]
            else:
                week_template = template[f"Hafta_{len(template)}"]

            plan["schedule"][week_key] = {}

            for day, exercises in week_template.items():
                if fitness_level < 3:
                    adjusted_exercises = exercises[:1]
                elif fitness_level < 7:
                    adjusted_exercises = exercises[:2]
                else:
                    adjusted_exercises = exercises

                plan["schedule"][week_key][day] = {
                    "exercises": adjusted_exercises,
                    "completed": False,
                    "date": None
                }

        self._save_plan(patient_name, plan)
        return plan

    def _save_plan(self, patient_name, plan):
        filename = self._plan_path(patient_name)
        self._atomic_save(filename, plan)
        print(f"✅ Plan kaydedildi: {filename}")

    def load_plan(self, patient_name):
        filename = self._plan_path(patient_name)
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def get_today_exercises(self, patient_name):
        plan = self.load_plan(patient_name)
        if not plan:
            return None

        today = datetime.now()
        start_date = datetime.fromisoformat(plan["start_date"])
        days_passed = (today - start_date).days

        current_week = (days_passed // 7) + 1
        if current_week > plan["weeks"]:
            return {"message": "Plan tamamlandı!", "exercises": []}

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
        start_date = datetime.fromisoformat(plan["created_date"])
        today = datetime.now()
        days_passed = (today - start_date).days
        return min((days_passed // 7) + 1, plan["weeks"])