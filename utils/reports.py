def summary_report(data):
    """Verilen kayıtlardan kısa bir özet oluşturur."""
    summary = {}
    for row in data:
        region = row["Bölge"]
        summary.setdefault(region, 0)
        summary[region] += 1
    return summary
