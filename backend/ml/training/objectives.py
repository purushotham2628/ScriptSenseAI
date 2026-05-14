from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


class SelfSupervisedObjectives:
    """Pretraining tasks that improve robustness before labels exist."""

    def masked_patch_loss(self, reconstructed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return ((reconstructed - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)

    def contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        anchor = nn.functional.normalize(anchor, dim=-1)
        positive = nn.functional.normalize(positive, dim=-1)
        logits = anchor @ positive.T / temperature
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return nn.functional.cross_entropy(logits, labels)


class DomainAdaptationLoss:
    """Encourages source and target script embeddings to share useful structure."""

    def coral_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = source - source.mean(dim=0, keepdim=True)
        target = target - target.mean(dim=0, keepdim=True)
        cs = source.T @ source / max(1, source.shape[0] - 1)
        ct = target.T @ target / max(1, target.shape[0] - 1)
        return ((cs - ct) ** 2).mean()


class FewShotEpisodeSampler:
    """Creates N-way K-shot training episodes for new scripts."""

    def build_episode(self, samples: List[Dict], n_way: int = 5, k_shot: int = 4) -> Tuple[List[Dict], List[Dict]]:
        by_label: Dict[str, List[Dict]] = {}
        for sample in samples:
            by_label.setdefault(sample.get("label", "unknown"), []).append(sample)
        support, query = [], []
        for label, label_samples in list(by_label.items())[:n_way]:
            support.extend(label_samples[:k_shot])
            query.extend(label_samples[k_shot: k_shot * 2])
        return support, query
