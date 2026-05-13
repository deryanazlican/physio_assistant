from core.progress_metrics import build_progress_payload
from core.progress_report import compare_progress_summaries

old_summary = build_progress_payload(
    exercise_code="ROM_ROT",
    reps=8,
    target_reps=10,
    done=False,
    movement_name="cervical_rotation",
    movement_value=5.0,
    movement_target=10.0,
    movement_unit="deg",
    quality_score=0.62,
    max_movement_value=5.0,
    right_value=5.0,
    left_value=9.0,
    right_reps=4,
    left_reps=4,
)

new_summary = build_progress_payload(
    exercise_code="ROM_ROT",
    reps=10,
    target_reps=10,
    done=True,
    movement_name="cervical_rotation",
    movement_value=8.0,
    movement_target=10.0,
    movement_unit="deg",
    quality_score=0.76,
    max_movement_value=8.0,
    right_value=8.0,
    left_value=10.0,
    right_reps=5,
    left_reps=5,
)

report = compare_progress_summaries(old_summary, new_summary)
print(report)