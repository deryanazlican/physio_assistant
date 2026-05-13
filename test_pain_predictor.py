from ai.pain_predictor import SimplePainPredictor

predictor = SimplePainPredictor()

current_data = {
    "exercise": "boyun_cevirme",
    "reps": 12,
    "duration": 8.5,
    "quality": 0.7,
    "current_pain": 3,
    "last_exercise_hours_ago": 30
}

history = [
    {"data": {"angle": 20}},
    {"data": {"angle": 24}},
    {"data": {"angle": 27}},
    {"data": {"angle": 30}},
]

prediction = predictor.predict_pain_after_exercise(current_data, history)

print(prediction)
print()
print(predictor.get_recommendation_text(prediction))
print()
print("Devam edilsin mi?", predictor.should_continue(prediction))