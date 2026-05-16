"""Backbone único EfficientNet-B0 (Opción A de la spec).

Sustituye la fusión de 3 backbones del Módulo 4 original. Devuelve un
vector 1280-d por imagen. Pesos congelados — ``requires_grad=False`` —
pero la cabeza del Módulo 5 se entrena sobre estas features.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Union

import numpy as np
import torch
from PIL import Image

from .config import CONFIG

logger = logging.getLogger(__name__)


class EfficientNetExtractor:
    """Wrapper sobre google/efficientnet-b0 (HuggingFace)."""

    def __init__(self,
                 model_id: str = None,
                 device: str = "cpu") -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.model_id = model_id or CONFIG["backbone_id"]
        self.device = torch.device(device)
        logger.info("Cargando backbone %s (esto descarga pesos la primera vez)…", self.model_id)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def extract(self, images: Union[Image.Image, Sequence[Image.Image]]) -> torch.Tensor:
        """Extrae features (B, 1280) para una o varias imágenes PIL."""
        if isinstance(images, Image.Image):
            images = [images]
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # EfficientNet retorna pooled_output con shape (B, 1280)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            # Fallback: promedio espacial sobre last_hidden_state
            lhs = outputs.last_hidden_state
            feats = lhs.mean(dim=tuple(range(2, lhs.ndim)))
        # Algunas versiones retornan (B, 1280, 1, 1)
        feats = feats.flatten(start_dim=1)
        return feats.detach().cpu()

    def extract_many(self, images: Iterable[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """Extrae features en batches para listas grandes."""
        chunks: List[torch.Tensor] = []
        batch: List[Image.Image] = []
        for img in images:
            batch.append(img)
            if len(batch) >= batch_size:
                chunks.append(self.extract(batch))
                batch = []
        if batch:
            chunks.append(self.extract(batch))
        return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, CONFIG["backbone_dim"]))
