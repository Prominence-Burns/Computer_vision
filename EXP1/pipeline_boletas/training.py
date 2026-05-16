"""M7 — Entrenamiento de la cabeza de clasificación.

Pipeline:
1. Toma un dataset sintético del Módulo 6.
2. Pasa cada imagen por el backbone congelado (Módulo 4) para obtener
   features 1280-d.
3. Splitea 70/15/15 con ``GroupShuffleSplit`` agrupando por
   ``source_image_id`` para evitar leakage entre augmentations.
4. Entrena la cabeza con AdamW + CosineAnnealingLR, BCEWithLogitsLoss y
   pos_weight calculado desde la distribución del split de train.
5. Persiste pesos + métricas.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from .classifier import ClassificationHead
from .config import CONFIG, PARTIDOS, coalition_to_onehot, ensure_outputs_dir
from .features import EfficientNetExtractor
from .synthetic import SyntheticSample

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: ClassificationHead
    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    test_indices: List[int]
    metrics_history: List[Dict[str, float]]


# ---------------------------------------------------------------------------
# Conversión dataset → tensores
# ---------------------------------------------------------------------------

def _samples_to_tensors(samples: List[SyntheticSample],
                        extractor: EfficientNetExtractor
                        ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Devuelve (features [N,1280], labels [N,9], group_ids)."""
    images = [s.image for s in samples]
    logger.info("Extrayendo features para %d muestras…", len(images))
    feats = extractor.extract_many(images, batch_size=16)
    labels = torch.stack([coalition_to_onehot(s.label_id) for s in samples])
    groups = [s.source_image_id for s in samples]
    return feats, labels, groups


def _split(features: torch.Tensor, labels: torch.Tensor,
           groups: List[str], seed: int = 42
           ) -> Tuple[List[int], List[int], List[int]]:
    n = len(features)
    idx = np.arange(n)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=CONFIG["split_val"] + CONFIG["split_test"],
                              random_state=seed)
    train_idx, holdout_idx = next(gss1.split(idx, groups=groups))
    holdout_groups = [groups[i] for i in holdout_idx]
    rel_test = CONFIG["split_test"] / (CONFIG["split_val"] + CONFIG["split_test"])
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed + 1)
    val_rel, test_rel = next(gss2.split(holdout_idx, groups=holdout_groups))
    val_idx = holdout_idx[val_rel].tolist()
    test_idx = holdout_idx[test_rel].tolist()
    return train_idx.tolist(), val_idx, test_idx


def _pos_weight(labels: torch.Tensor) -> torch.Tensor:
    n_pos = labels.sum(dim=0)
    n_neg = labels.shape[0] - n_pos
    return n_neg / torch.clamp(n_pos, min=1.0)


# ---------------------------------------------------------------------------
# Loop de entrenamiento
# ---------------------------------------------------------------------------

def train_head(samples: List[SyntheticSample],
               extractor: EfficientNetExtractor,
               num_epochs: int = None,
               batch_size: int = None,
               lr: float = None,
               device: str = "cpu") -> TrainingResult:
    num_epochs = num_epochs or CONFIG["num_epochs"]
    batch_size = batch_size or CONFIG["batch_size"]
    lr = lr or CONFIG["lr"]

    feats, labels, groups = _samples_to_tensors(samples, extractor)
    train_idx, val_idx, test_idx = _split(feats, labels, groups, seed=CONFIG["seed"])
    logger.info("Split sizes — train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    train_x, train_y = feats[train_idx], labels[train_idx]
    val_x, val_y = feats[val_idx], labels[val_idx]
    test_x, test_y = feats[test_idx], labels[test_idx]

    pos_w = _pos_weight(train_y).to(device)
    model = ClassificationHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True,
    )

    history: List[Dict[str, float]] = []
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()

        # Validación
        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x.to(device)).cpu()
            val_pred = (torch.sigmoid(val_logits) >= CONFIG["decision_threshold"]).int().numpy()
            val_true = val_y.int().numpy()
            val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
            val_hamming = hamming_loss(val_true, val_pred)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "val_f1_macro": float(val_f1),
            "val_hamming": float(val_hamming),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_metrics)
        logger.info("Epoch %d/%d  loss=%.4f  val_f1_macro=%.3f  val_hamming=%.3f",
                    epoch, num_epochs, epoch_metrics["train_loss"],
                    val_f1, val_hamming)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                logger.info("Early stopping en epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Persistir modelo + métricas
    out_dir = ensure_outputs_dir()
    torch.save(model.state_dict(), out_dir / "model_head.pt")
    with (out_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"history": history,
                   "best_val_f1_macro": best_val_f1,
                   "n_train": len(train_idx),
                   "n_val": len(val_idx),
                   "n_test": len(test_idx)}, f, indent=2, ensure_ascii=False)

    return TrainingResult(
        model=model,
        train_features=train_x, train_labels=train_y,
        val_features=val_x, val_labels=val_y,
        test_features=test_x, test_labels=test_y,
        test_indices=test_idx,
        metrics_history=history,
    )


def load_trained_head(path: Path | str = None, device: str = "cpu") -> ClassificationHead:
    path = Path(path) if path else CONFIG["model_path"]
    model = ClassificationHead()
    if Path(path).exists():
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        logger.info("Modelo cargado de %s", path)
    else:
        logger.warning("No hay modelo entrenado en %s — devolviendo cabeza inicializada", path)
    model.to(device).eval()
    return model
