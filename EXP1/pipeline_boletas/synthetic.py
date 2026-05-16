"""M6 — Generador sintético de boletas.

Dibuja boletas ficticias con PIL respetando la estructura mínima necesaria
para que M2 (detección de marcas) localice 11 recuadros y clasifique las
marcas según el ``label`` esperado. La salida es un dataset de tuplas
``(PIL.Image, marcas_gt, label_id, source_image_id)`` que alimenta el
entrenamiento de la cabeza neuronal y la evaluación.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import (
    CATALOGO_RECUADROS,
    COALICIONES_2024,
    CONFIG,
    DISTRIBUCION_ETIQUETAS,
    PARTIDOS,
    TipoBoleta,
)


# ---------------------------------------------------------------------------
# Layout: distribución fija de los 11 recuadros sobre una boleta retrato
# ---------------------------------------------------------------------------

# Cuadrícula 4 columnas × 3 filas (12 celdas, usamos las primeras 11 en orden
# del catálogo). Coordenadas relativas (x0, y0, x1, y1) sobre [0, 1].
_GRID_COLS = 4
_GRID_ROWS = 3
_HEADER_FRAC = 0.18    # 18% superior reservado para encabezado
_FOOTER_FRAC = 0.05


def _grid_layout(image_size: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Devuelve [(recuadro_id, (x0,y0,x1,y1)), ...] en píxeles."""
    W, H = image_size
    top = int(H * _HEADER_FRAC)
    bottom = int(H * (1 - _FOOTER_FRAC))
    cell_w = W // _GRID_COLS
    cell_h = (bottom - top) // _GRID_ROWS

    layout = []
    for idx, recuadro in enumerate(CATALOGO_RECUADROS):
        col = idx % _GRID_COLS
        row = idx // _GRID_COLS
        x0 = col * cell_w + 12
        y0 = top + row * cell_h + 12
        x1 = (col + 1) * cell_w - 12
        y1 = top + (row + 1) * cell_h - 12
        layout.append((recuadro, (x0, y0, x1, y1)))
    return layout


@dataclass
class MarcaGT:
    recuadro: str
    tipo_marca: str
    intensidad: str
    dentro_recuadro: bool
    proporcion_en_recuadro: float


@dataclass
class SyntheticSample:
    image: Image.Image
    marcas_gt: List[MarcaGT]
    label_id: str                     # ID destinatario (partido / coalición / NULO)
    tipo_boleta: TipoBoleta
    source_image_id: str
    boxes: List[Tuple[str, Tuple[int, int, int, int]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Resolución de etiqueta → recuadros a marcar
# ---------------------------------------------------------------------------

def _recuadros_a_marcar(label_id: str) -> List[str]:
    """Dado un destinatario, devuelve los recuadros donde el elector marcaría."""
    if label_id in COALICIONES_2024:
        return list(COALICIONES_2024[label_id]["partidos"])
    if label_id in PARTIDOS and label_id != "NULO":
        return [label_id]
    if label_id == "NULO":
        return []   # voto en blanco / sin marcas válidas
    raise ValueError(f"label_id desconocido: {label_id}")


# ---------------------------------------------------------------------------
# Dibujo de marcas
# ---------------------------------------------------------------------------

_TIPOS_MARCA = ["cruz", "raya", "palomita", "circulo"]


def _draw_mark(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int],
               tipo: str, color: Tuple[int, int, int], width: int = 4) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = min(x1 - x0, y1 - y0) // 3

    if tipo == "cruz":
        draw.line([(cx - half, cy - half), (cx + half, cy + half)], fill=color, width=width)
        draw.line([(cx - half, cy + half), (cx + half, cy - half)], fill=color, width=width)
    elif tipo == "raya":
        draw.line([(cx - half, cy), (cx + half, cy)], fill=color, width=width)
    elif tipo == "palomita":
        draw.line([(cx - half, cy), (cx - half // 3, cy + half)], fill=color, width=width)
        draw.line([(cx - half // 3, cy + half), (cx + half, cy - half)], fill=color, width=width)
    elif tipo == "circulo":
        draw.ellipse((cx - half, cy - half, cx + half, cy + half), outline=color, width=width)


def _draw_recuadros(draw: ImageDraw.ImageDraw,
                    layout: List[Tuple[str, Tuple[int, int, int, int]]],
                    font: ImageFont.ImageFont) -> None:
    for recuadro, box in layout:
        draw.rectangle(box, outline=(0, 0, 0), width=2)
        # etiqueta del partido/opción dentro del recuadro
        draw.text((box[0] + 6, box[1] + 4), recuadro, fill=(0, 0, 0), font=font)


def _draw_header(draw: ImageDraw.ImageDraw, image_size: Tuple[int, int],
                 tipo_boleta: TipoBoleta, font_h: ImageFont.ImageFont) -> None:
    W, _ = image_size
    draw.text((20, 20), "INSTITUTO NACIONAL ELECTORAL", fill=(0, 0, 0), font=font_h)
    draw.text((20, 60), f"BOLETA — ELECCIÓN DE {tipo_boleta.value.upper()}", fill=(0, 0, 0), font=font_h)
    draw.line([(10, 110), (W - 10, 110)], fill=(0, 0, 0), width=2)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Carga una fuente legible; cae a default de PIL si no hay TrueType."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Augmentations sobre imagen
# ---------------------------------------------------------------------------

def _apply_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    """Augmentation que preserva geometría (los ``boxes_hint`` siguen siendo válidos).

    En el MVP los recuadros se conocen por layout (no por detección visual),
    así que NO aplicamos rotación/crop/flip para mantener la alineación.
    Las transformaciones fotométricas (brillo, blur, ruido) sí simulan
    condiciones de captura con celular sin romper la geometría.
    """
    # Brillo / contraste vía punto
    if rng.random() < 0.7:
        factor = rng.uniform(0.70, 1.25)
        img = Image.eval(img, lambda v: int(min(255, max(0, v * factor))))
    # Blur gaussiano
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.6)))
    # Ruido sal y pimienta
    if rng.random() < 0.6:
        arr = np.array(img)
        n_noise = int(arr.size * 0.002)
        for _ in range(n_noise):
            x = rng.randrange(arr.shape[1])
            y = rng.randrange(arr.shape[0])
            arr[y, x] = 0 if rng.random() < 0.5 else 255
        img = Image.fromarray(arr)
    # Sombra simulada (gradiente vertical multiplicativo)
    if rng.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        h = arr.shape[0]
        gradient = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.6, 0.95), h)
        arr *= gradient[:, None, None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


# ---------------------------------------------------------------------------
# Sampling de etiquetas
# ---------------------------------------------------------------------------

def _sample_label(rng: random.Random) -> str:
    keys = list(DISTRIBUCION_ETIQUETAS.keys())
    weights = [DISTRIBUCION_ETIQUETAS[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------

class SyntheticBallotGenerator:
    """Crea boletas sintéticas reproducibles a partir del CONFIG."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.image_size: Tuple[int, int] = CONFIG["synthetic_image_size"]
        self.rng = random.Random(seed if seed is not None else CONFIG["seed"])
        self._font_label = _load_font(14)
        self._font_header = _load_font(18)

    # ---- API de alto nivel -------------------------------------------------

    def build_dataset(
        self,
        n_sources: Optional[int] = None,
        n_aug_per_source: Optional[int] = None,
        tipo_boleta: TipoBoleta = TipoBoleta.DIPUTACION_MR,
    ) -> List[SyntheticSample]:
        """Genera ``n_sources`` boletas base y por cada una ``n_aug+1`` muestras."""
        n_sources = n_sources or CONFIG["synthetic_n_source_images"]
        n_aug = n_aug_per_source if n_aug_per_source is not None else CONFIG["synthetic_n_augmentations"]

        samples: List[SyntheticSample] = []
        for src_idx in range(n_sources):
            label = _sample_label(self.rng)
            base = self._render_base(label, tipo_boleta)
            source_id = f"src-{src_idx:04d}-{label}"
            base.source_image_id = source_id
            samples.append(base)
            for aug_idx in range(n_aug):
                aug_img = _apply_augmentation(base.image.copy(), self.rng)
                samples.append(SyntheticSample(
                    image=aug_img,
                    marcas_gt=base.marcas_gt,
                    label_id=label,
                    tipo_boleta=tipo_boleta,
                    source_image_id=source_id,
                    boxes=base.boxes,
                ))
        return samples

    # ---- Render de una boleta limpia --------------------------------------

    def _render_base(self, label_id: str, tipo_boleta: TipoBoleta) -> SyntheticSample:
        W, H = self.image_size
        img = Image.new("RGB", (W, H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        _draw_header(draw, (W, H), tipo_boleta, self._font_header)
        layout = _grid_layout((W, H))
        _draw_recuadros(draw, layout, self._font_label)

        marcas_gt: List[MarcaGT] = []
        recuadros_objetivo = _recuadros_a_marcar(label_id)
        # Asignamos un tipo de marca aleatorio para todos los recuadros objetivo
        tipo = self.rng.choice(_TIPOS_MARCA)
        for recuadro, box in layout:
            if recuadro not in recuadros_objetivo:
                continue
            color = (10, 10, 10)
            _draw_mark(draw, box, tipo, color, width=5)
            marcas_gt.append(MarcaGT(
                recuadro=recuadro,
                tipo_marca=tipo,
                intensidad="clara",
                dentro_recuadro=True,
                proporcion_en_recuadro=0.85,
            ))

        return SyntheticSample(
            image=img,
            marcas_gt=marcas_gt,
            label_id=label_id,
            tipo_boleta=tipo_boleta,
            source_image_id="",
            boxes=layout,
        )


def render_single_demo_ballot(label_id: str = "SHH",
                              tipo_boleta: TipoBoleta = TipoBoleta.DIPUTACION_MR) -> SyntheticSample:
    """Helper para generar UNA boleta limpia (para `main classify` demo)."""
    gen = SyntheticBallotGenerator()
    sample = gen._render_base(label_id, tipo_boleta)
    sample.source_image_id = f"demo-{label_id}"
    return sample
