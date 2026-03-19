from __future__ import annotations

from typing import Callable

from src.models.scoring_model import BaseModel, RuleBasedModel


ModelFactory = Callable[[], BaseModel]


class ModelRegistry:
    def __init__(self):
        self._registry: dict[str, ModelFactory] = {
            "xgboost": self._create_xgboost,
            "logistic": self._create_logistic,
            "rules": RuleBasedModel,
        }

    def list_models(self) -> list[str]:
        return sorted(self._registry.keys())

    def create(self, model_name: str) -> BaseModel:
        key = model_name.strip().lower()
        if key not in self._registry:
            raise ValueError(f"Unknown model: {model_name}")
        return self._registry[key]()

    def create_candidates(self, preferred: list[str] | None = None) -> list[tuple[str, BaseModel]]:
        names = preferred or self.list_models()
        return [(name, self.create(name)) for name in names]

    @staticmethod
    def _create_xgboost() -> BaseModel:
        from src.models.xgb_model import XGBoostModel

        return XGBoostModel(model_path="data/xgb_model.json")

    @staticmethod
    def _create_logistic() -> BaseModel:
        from src.models.logistic_model import LogisticModel

        return LogisticModel(model_path="data/logistic_model.pkl")
