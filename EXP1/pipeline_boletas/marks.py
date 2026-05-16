"""M2 — Detección y caracterización de marcas por recuadro.

Implementa dos modos:

1. **boxes_hint**: si se conocen los recuadros (caso del demo sintético, o
   cuando otro módulo provee el layout), evita la detección por contornos
   y va directo al análisis intra-recuadro.
2. **detección por contornos**: para imágenes reales, busca la cuadrícula
   con ``cv2.findContours`` filtrando por aspect ratio y área.

Para cada recuadro detectado clasifica la marca con momentos de Hu
(``cv2.HuMoments``) y mide intensidad + posición.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import CATALOGO_RECUADROS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estructura de salida (alineada con el JSON AECC)
# ---------------------------------------------------------------------------

@dataclass
class MarcaDetectada:
    recuadro: str
    tipo_marca: str
    intensidad: str
    dentro_recuadro: bool
    proporcion_en_recuadro: float
    # Campos auxiliares (no se serializan al JSON AECC):
    bbox_recuadro: Tuple[int, int, int, int] = (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Clasificación de tipo de marca
# ---------------------------------------------------------------------------

_TIPOS_VALIDOS = ("cruz", "raya", "palomita", "circulo", "texto", "mancha", "reflejo", "otro")


def _classify_mark_type(crop_bin: np.ndarray) -> str:
    """Clasifica el tipo de marca dentro de un recuadro binarizado.

    Heurísticas simples basadas en momentos y forma del contorno principal:
    - Sin pixeles → ningún tipo (caller debe decidir).
    - aspect ratio alto y bajo grosor → ``raya``.
    - 2 ramas que se cruzan en el bbox → ``cruz``.
    - contorno cerrado con extent bajo → ``circulo``.
    - asimetría diagonal → ``palomita``.
    - área grande sin patrón → ``mancha``.
    """
    h, w = crop_bin.shape[:2]
    n_marca = int(np.sum(crop_bin > 0))
    if n_marca < 10:
        return "otro"

    contornos, _ = cv2.findContours(crop_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return "otro"

    contorno = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(contorno)
    if area < 8:
        return "otro"

    x, y, bw, bh = cv2.boundingRect(contorno)
    aspect = bw / max(bh, 1)
    extent = area / max(bw * bh, 1)
    perim = cv2.arcLength(contorno, True)
    circularity = (4 * np.pi * area) / (perim * perim) if perim > 0 else 0.0

    # Reflejo: brillo uniforme cubriendo la mayor parte (lo manejamos antes
    # de binarizar; aquí no podemos detectarlo). Caller puede pre-clasificar.

    if circularity > 0.55 and extent < 0.5:
        return "circulo"
    if aspect > 2.5 and bh < h * 0.30:
        return "raya"

    # Para distinguir cruz vs palomita observamos cuántos cuadrantes del
    # bounding-box del contorno tienen píxeles negros (la cruz cubre los 4,
    # la palomita 2-3).
    sub = crop_bin[y:y + bh, x:x + bw]
    if sub.size == 0:
        return "otro"
    sh, sw = sub.shape
    q1 = sub[:sh // 2, :sw // 2].sum()
    q2 = sub[:sh // 2, sw // 2:].sum()
    q3 = sub[sh // 2:, :sw // 2].sum()
    q4 = sub[sh // 2:, sw // 2:].sum()
    quadrants_with_ink = sum(1 for q in (q1, q2, q3, q4) if q > 0)

    if quadrants_with_ink >= 4 and extent > 0.10:
        return "cruz"
    if quadrants_with_ink >= 2 and aspect > 0.5:
        return "palomita"
    if extent > 0.55:
        return "mancha"
    return "otro"


def _classify_intensity(crop_gray: np.ndarray, mask: np.ndarray) -> str:
    """Clasifica intensidad por contraste medio marca-vs-fondo."""
    if mask.sum() == 0:
        return "borrosa"
    vals_marca = crop_gray[mask > 0]
    vals_fondo = crop_gray[mask == 0]
    if vals_fondo.size == 0:
        return "clara"
    contrast = abs(float(np.mean(vals_fondo)) - float(np.mean(vals_marca))) / 255.0
    if contrast >= 0.40:
        return "clara"
    if contrast >= 0.20:
        return "tenue"
    return "borrosa"


# ---------------------------------------------------------------------------
# Detección de cuadrícula
# ---------------------------------------------------------------------------

def _detect_recuadros(img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Devuelve bounding-boxes (x, y, w, h) de candidatos a recuadro.

    Filtra por área y aspect ratio (los recuadros son aprox. cuadrados o
    ligeramente apaisados). En boletas reales, complementar este filtro con
    posición vertical (encabezado vs grid).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bin_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10,
    )
    contornos, _ = cv2.findContours(bin_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    img_area = h_img * w_img
    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 0.005 * img_area or area > 0.20 * img_area:
            continue
        ar = w / max(h, 1)
        if ar < 0.4 or ar > 3.5:
            continue
        cajas.append((x, y, w, h))
    return cajas


# ---------------------------------------------------------------------------
# Módulo principal
# ---------------------------------------------------------------------------

class MarkDetectionModule:
    """Detecta marcas por recuadro.

    Si ``boxes_hint`` viene, se asume que los 11 recuadros del catálogo
    AECC ocupan exactamente esas cajas (mismo orden que ``CATALOGO_RECUADROS``).
    """

    def __init__(self) -> None:
        pass

    def detect(self,
               img_rgb: np.ndarray,
               boxes_hint: Optional[List[Tuple[str, Tuple[int, int, int, int]]]] = None
               ) -> List[MarcaDetectada]:
        if boxes_hint is None:
            boxes = _detect_recuadros(img_rgb)
            # Mapeo posicional al catálogo (orden top→bottom, left→right):
            boxes = sorted(boxes, key=lambda b: (b[1] // 60, b[0]))
            boxes = boxes[: len(CATALOGO_RECUADROS)]
            boxes_named = [
                (CATALOGO_RECUADROS[i], (x, y, x + w, y + h))
                for i, (x, y, w, h) in enumerate(boxes)
            ]
        else:
            boxes_named = boxes_hint

        marcas: List[MarcaDetectada] = []
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        for recuadro_id, (x0, y0, x1, y1) in boxes_named:
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(img_rgb.shape[1], x1), min(img_rgb.shape[0], y1)
            if x1c - x0c < 5 or y1c - y0c < 5:
                continue
            crop_gray = gray[y0c:y1c, x0c:x1c]
            # Binariza pintura interna (la línea del propio recuadro la
            # eliminamos haciendo erosión leve antes).
            _, bin_crop = cv2.threshold(crop_gray, 0, 255,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Borra el contorno externo del recuadro: anula los píxeles a 4 px del borde
            margin = 4
            inner = np.zeros_like(bin_crop)
            inner[margin:-margin, margin:-margin] = bin_crop[margin:-margin, margin:-margin]

            # Quita la etiqueta de texto del partido (esquina superior izquierda):
            # mascara proporcional — top 22% × left 50% del crop.
            ch, cw = inner.shape
            label_h = max(8, int(ch * 0.22))
            label_w = max(20, int(cw * 0.50))
            inner[:label_h, :label_w] = 0

            n_marca = int((inner > 0).sum())
            min_area = max(40, int(0.005 * ch * cw))
            if n_marca < min_area:
                continue   # recuadro vacío → no se reporta

            tipo = _classify_mark_type(inner)
            intensidad = _classify_intensity(crop_gray, inner)

            # Centroide de la marca
            ys, xs = np.where(inner > 0)
            cx = float(xs.mean()) + x0c
            cy = float(ys.mean()) + y0c

            dentro = (x0c <= cx <= x1c) and (y0c <= cy <= y1c)
            # Proporción: con boxes_hint, todo el inner está dentro → ~1.0
            proporcion = 1.0 if dentro else 0.0
            # Ajuste fino: cuántos píxeles del inner caen dentro del recuadro
            # (siempre 100% en este crop; queda por mejorar para detección real).

            marcas.append(MarcaDetectada(
                recuadro=recuadro_id,
                tipo_marca=tipo,
                intensidad=intensidad,
                dentro_recuadro=bool(dentro),
                proporcion_en_recuadro=float(proporcion),
                bbox_recuadro=(x0, y0, x1, y1),
            ))
        return marcas
