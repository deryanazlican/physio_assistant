from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class ExerciseCase:
    case_id: str
    exercise_name: str
    patient_note: str
    metrics: Dict[str, Any]
    expected_focus: str


@dataclass
class ModelResponse:
    model_name: str
    raw_text: str
    latency_seconds: float
    success: bool
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)