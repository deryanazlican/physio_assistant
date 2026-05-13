from abc import ABC, abstractmethod
from .schemas import ExerciseCase, ModelResponse


class BaseLLMClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_feedback(self, case: ExerciseCase) -> ModelResponse:
        pass