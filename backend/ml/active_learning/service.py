from dataclasses import dataclass
from typing import Dict, List

from backend.schemas.platform import AnnotationCorrection, PredictionResponse


@dataclass
class ActiveLearningDecision:
    should_review: bool
    priority: float
    reasons: List[str]


class ActiveLearningService:
    """Selects uncertain/unseen samples for human correction and continual learning."""

    def score_prediction(self, prediction: PredictionResponse) -> ActiveLearningDecision:
        reasons: List[str] = []
        priority = 0.0
        if prediction.confidence < 0.72:
            priority += 0.45
            reasons.append("low_confidence")
        if prediction.anomaly_score > 0.45:
            priority += 0.35
            reasons.append("unseen_script_anomaly")
        if prediction.unknown_symbol_ratio > 0.12:
            priority += 0.25
            reasons.append("many_unknown_symbols")
        return ActiveLearningDecision(should_review=priority > 0.35, priority=min(priority, 1.0), reasons=reasons)

    def build_training_sample(self, correction: AnnotationCorrection) -> Dict:
        return {
            "prediction_id": correction.prediction_id,
            "text": correction.corrected_text,
            "symbols": correction.corrected_symbols,
            "source": "human_correction",
            "continual_learning_weight": 1.5,
            "notes": correction.notes,
        }

    def replay_buffer_policy(self, new_samples: List[Dict], old_samples: List[Dict], max_size: int = 50000) -> List[Dict]:
        # Keeps a mixture of new domain samples and older exemplars to reduce catastrophic forgetting.
        old_quota = max_size // 2
        new_quota = max_size - old_quota
        return old_samples[-old_quota:] + new_samples[-new_quota:]
