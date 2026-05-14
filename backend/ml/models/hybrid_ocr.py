from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn


@dataclass
class HybridPrediction:
    text: str
    confidence: float
    token_confidences: List[float]
    unknown_symbol_ratio: float
    embeddings: np.ndarray
    metadata: Dict


class VisionEncoder(nn.Module):
    """Placeholder interface for ViT/Swin/ConvNeXt encoders.

    In production, replace the small CNN with a timm/HuggingFace backbone such as
    SwinV2, ConvNeXtV2, or BEiT pretrained through self-supervised manuscript tasks.
    """

    def __init__(self, embedding_dim: int = 768) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(), nn.Linear(128 * 8 * 8, embedding_dim), nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class HybridAncientOCR(nn.Module):
    """Hybrid recognizer: vision encoder + sequence decoder + OCR head.

    The intended production variants are:
    - Vision: Swin/ViT/ConvNeXt with self-supervised pretraining.
    - Sequence: Transformer decoder or BiLSTM+CTC for low-latency OCR.
    - Context: HuggingFace/GPT-style correction model after OCR.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 768, hidden_dim: int = 384) -> None:
        super().__init__()
        self.vision = VisionEncoder(embedding_dim=embedding_dim)
        self.sequence = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8, batch_first=True),
            num_layers=4,
        )
        self.query_tokens = nn.Parameter(torch.randn(1, 96, embedding_dim) * 0.02)
        self.ocr_head = nn.Linear(embedding_dim, vocab_size)
        self.embedding_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual = self.vision(image)
        memory = visual.unsqueeze(1)
        queries = self.query_tokens.repeat(image.shape[0], 1, 1)
        decoded = self.sequence(queries, memory)
        logits = self.ocr_head(decoded)
        embedding = self.embedding_projection(visual)
        return {"logits": logits, "embedding": embedding}


class DynamicLabelMapper:
    """Allows new scripts to introduce new symbols without rebuilding everything."""

    def __init__(self) -> None:
        self.symbol_to_id = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.id_to_symbol = {idx: sym for sym, idx in self.symbol_to_id.items()}

    def fit_incremental(self, symbols: List[str]) -> None:
        for symbol in symbols:
            if symbol not in self.symbol_to_id:
                idx = len(self.symbol_to_id)
                self.symbol_to_id[symbol] = idx
                self.id_to_symbol[idx] = symbol

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_symbol.get(idx, "<unk>") for idx in ids]
        return "".join(token for token in tokens if token not in {"<pad>", "<bos>", "<eos>"})

    @property
    def vocab_size(self) -> int:
        return len(self.symbol_to_id)
