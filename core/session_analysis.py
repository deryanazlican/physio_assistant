def classify_session(pain_after):
    if pain_after < 4:
        return "good"
    elif pain_after < 7:
        return "medium"
    else:
        return "risky"