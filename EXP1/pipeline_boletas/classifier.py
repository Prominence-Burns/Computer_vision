"""M5.5 — Cabeza de clasificación multi-etiqueta + masking por tipo de boleta.

La cabeza recibe el vector 1280-d del backbone y emite logits sobre 9 clases
(orden de ``PARTIDOS``). NO aplica sigmoide internamente — la pérdida es
``BCEWithLogitsLoss`` y aplicamos ``torch.sigmoid`` sólo en inferencia.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import CLASES_POR_BOLETA, CONFIG, PARTIDOS, TipoBoleta


class ClassificationHead(nn.Module):
    """Cabeza ligera: LayerNorm → Linear(d → h) → ReLU → Dropout → Linear(h → C)."""

    def __init__(self,
                 input_dim: int = None,
                 hidden_dim: int = None,
                 n_classes: int = None,
                 dropout: float = None) -> None:
        super().__init__()
        input_dim = input_dim or CONFIG["backbone_dim"]
        hidden_dim = hidden_dim or CONFIG["head_hidden_dim"]
        n_classes = n_classes or CONFIG["n_classes"]
        dropout = dropout if dropout is not None else CONFIG["head_dropout"]

        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.relu(self.fc1(self.norm(x)))))


def aplicar_mascara_tipo_boleta(logits: torch.Tensor, tipo_boleta: TipoBoleta) -> torch.Tensor:
    """Pone -inf en clases inactivas para el tipo de boleta detectado."""
    if tipo_boleta not in CLASES_POR_BOLETA:
        return logits
    activas = set(CLASES_POR_BOLETA[tipo_boleta])
    mask = torch.tensor(
        [0.0 if p in activas else float("-inf") for p in PARTIDOS],
        device=logits.device, dtype=logits.dtype,
    )
    return logits + mask


def infer(model: ClassificationHead,
          features: torch.Tensor,
          tipo_boleta: TipoBoleta = TipoBoleta.DIPUTACION_MR
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (logits enmascarados, probabilidades sigmoideas)."""
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        masked = aplicar_mascara_tipo_boleta(logits, tipo_boleta)
        probs = torch.sigmoid(masked)
    return masked, probs
