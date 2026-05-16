"""M8 — Métricas de evaluación.

Métricas primarias (lo que importa para el AECC):
- exact destinatario accuracy (post R1–R5)
- F1 macro / micro sobre 9 clases
- confusion matrix por subtipo
- tasa requiere_revision

Secundarias (diagnóstico del clasificador): hamming, LRAP, subset acc,
coverage error.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
)

from .config import CONFIG, PARTIDOS, ensure_outputs_dir

logger = logging.getLogger(__name__)


def _safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Métrica %s falló: %s", getattr(fn, "__name__", fn), exc)
        return 0.0


def evaluate_classifier(y_true: torch.Tensor, y_pred: torch.Tensor,
                        y_scores: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Métricas neuronales puras (sin reglas R1–R5)."""
    y_true_np = y_true.int().numpy()
    y_pred_np = y_pred.int().numpy()
    metrics = {
        "f1_macro": _safe_metric(f1_score, y_true_np, y_pred_np, average="macro", zero_division=0),
        "f1_micro": _safe_metric(f1_score, y_true_np, y_pred_np, average="micro", zero_division=0),
        "hamming_loss": _safe_metric(hamming_loss, y_true_np, y_pred_np),
        "subset_accuracy": _safe_metric(accuracy_score, y_true_np, y_pred_np),
    }
    if y_scores is not None:
        scores_np = y_scores.numpy()
        metrics["lrap"] = _safe_metric(label_ranking_average_precision_score, y_true_np, scores_np)
        metrics["coverage_error"] = _safe_metric(coverage_error, y_true_np, scores_np)
    return metrics


def evaluate_destinatarios(destinatarios_true: Sequence[Optional[str]],
                            destinatarios_pred: Sequence[Optional[str]]) -> Dict[str, float]:
    """Exact destinatario accuracy y matriz de confusión rugosa."""
    n = len(destinatarios_true)
    if n == 0:
        return {"exact_destinatario_accuracy": 0.0, "n": 0}
    correct = sum(1 for t, p in zip(destinatarios_true, destinatarios_pred) if t == p)
    return {
        "exact_destinatario_accuracy": correct / n,
        "n": n,
    }


def confusion_subtipos(true_subtipos: Sequence[str],
                       pred_subtipos: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Devuelve dict[true][pred] → conteo."""
    matrix: Dict[str, Counter] = {}
    for t, p in zip(true_subtipos, pred_subtipos):
        matrix.setdefault(t, Counter())[p] += 1
    return {t: dict(counts) for t, counts in matrix.items()}


def tasa_requiere_revision(flags: Sequence[bool]) -> float:
    if not flags:
        return 0.0
    return sum(1 for f in flags if f) / len(flags)


def health_check_sintetico(emb_sint: torch.Tensor, emb_real: torch.Tensor) -> float:
    """Distancia coseno media centroide-sintético ↔ embeddings reales.

    Devuelve 0.0 si no hay embeddings reales (carpeta seed vacía).
    """
    import torch.nn.functional as F
    if emb_real.numel() == 0:
        return 0.0
    centroide = emb_sint.mean(dim=0, keepdim=True)
    return float((1 - F.cosine_similarity(centroide, emb_real)).mean().item())


def write_evaluation_report(report: Dict, path: Optional[Path] = None) -> Path:
    out_dir = ensure_outputs_dir()
    target = Path(path) if path else out_dir / "eval_report.json"
    with target.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return target
