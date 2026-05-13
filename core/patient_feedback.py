from typing import Dict, Any, List, Optional


def _safe_num(value, default=0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def build_patient_feedback(report: Dict[str, Any], new_summary: Dict[str, Any]) -> str:
    movement_change = _safe_num(report.get("movement_change"))
    reps_change = int(report.get("reps_change", 0) or 0)
    quality_change = _safe_num(report.get("quality_change"))
    symmetry_change = report.get("symmetry_change")
    pain_change = report.get("pain_change")

    movement_name = str(new_summary.get("movement_name", "hareket"))
    unit = str(new_summary.get("movement_unit", ""))
    new_max = _safe_num(report.get("new_max_movement"))
    old_max = _safe_num(report.get("old_max_movement"))

    lines: List[str] = []

    # Genel durum
    positive_score = 0
    negative_score = 0

    if movement_change > 0:
        positive_score += 1
    elif movement_change < 0:
        negative_score += 1

    if reps_change > 0:
        positive_score += 1
    elif reps_change < 0:
        negative_score += 1

    if quality_change > 0:
        positive_score += 1
    elif quality_change < 0:
        negative_score += 1

    if pain_change is not None:
        if pain_change < 0:
            positive_score += 1
        elif pain_change > 0:
            negative_score += 1

    if positive_score > negative_score:
        lines.append("Önceki seansınıza göre daha iyi durumdasınız.")
    elif negative_score > positive_score:
        lines.append("Bu seansta bazı alanlarda zorlanma görülüyor.")
    else:
        lines.append("Önceki seansınıza benzer bir performans gösterdiniz.")

    # Hareket açıklığı
    if movement_change > 0:
        lines.append(
            f"Hareket açıklığınız arttı. Önceki en yüksek değeriniz {old_max:.2f} iken bu seansta {new_max:.2f} oldu."
        )
    elif movement_change < 0:
        lines.append(
            f"Hareket açıklığınız bir miktar azaldı. Önceki en yüksek değeriniz {old_max:.2f}, bu seansta {new_max:.2f}."
        )
    else:
        lines.append("Hareket açıklığınız önceki seansla benzer seviyede kaldı.")

    # Tekrar
    if reps_change > 0:
        lines.append("Tamamladığınız tekrar sayısı arttı.")
    elif reps_change < 0:
        lines.append("Tamamladığınız tekrar sayısı azaldı.")
    else:
        lines.append("Tekrar sayınız önceki seansla aynı kaldı.")

    # Kalite
    if quality_change > 0.03:
        lines.append("Hareket kalitenizde iyileşme var.")
    elif quality_change < -0.03:
        lines.append("Hareket kalitenizde bir miktar düşüş var.")

    # Simetri
    if symmetry_change is not None:
        symmetry_change = _safe_num(symmetry_change)
        if symmetry_change > 0.03:
            lines.append("Sağ ve sol taraf arasındaki denge daha iyi hale gelmiş.")
        elif symmetry_change < -0.03:
            lines.append("Sağ ve sol taraf arasında hâlâ fark var. İki tarafı dengeli çalıştırmaya devam edin.")

    # Ağrı
    if pain_change is not None:
        pain_change = _safe_num(pain_change)
        if pain_change < 0:
            lines.append("Egzersiz sonrası ağrı seviyeniz azalmış görünüyor.")
        elif pain_change > 0:
            lines.append("Egzersiz sonrası ağrı seviyeniz artmış görünüyor. Dikkatli ilerleyin.")

    # Son cümle
    if movement_change > 0 and (symmetry_change is None or _safe_num(symmetry_change) >= -0.03):
        lines.append("Bu şekilde kontrollü devam edebilirsiniz.")
    else:
        lines.append("Daha kontrollü ve dengeli tekrarlarla devam etmeniz faydalı olacaktır.")

    return " ".join(lines)