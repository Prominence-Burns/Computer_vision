"""M1 — Preprocesamiento + OCR.

Devuelve los dos artefactos exigidos por la spec:
- ``imagen_alta_res``: ndarray uint8 RGB con lado largo ≥ 1600 px.
- ``imagen_backbone``: PIL.Image 224×224 RGB para el extractor de features.

Más texto OCR + bounding boxes (cuando ``pytesseract`` está disponible).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estructuras de salida
# ---------------------------------------------------------------------------

@dataclass
class RegionOCR:
    text: str
    box: Tuple[int, int, int, int]   # (x, y, w, h)
    conf: float


@dataclass
class PreprocessOutput:
    imagen_alta_res: np.ndarray              # uint8 RGB, ≥1600 px lado largo
    imagen_backbone: Image.Image             # PIL 224×224 RGB
    texto_ocr: str
    regiones_ocr: List[RegionOCR] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers de geometría
# ---------------------------------------------------------------------------

def _resize_long_edge(img: np.ndarray, target_long: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge >= target_long:
        return img
    scale = target_long / long_edge
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _try_deskew(img: np.ndarray) -> np.ndarray:
    """Intenta corrección de perspectiva con findContours; tolera fallos."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        # Mayor contorno con 4 vértices después de aproximación
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            return img
        h_img, w_img = img.shape[:2]
        if cv2.contourArea(approx) < 0.30 * (h_img * w_img):
            return img  # contorno demasiado pequeño, probablemente no es la boleta
        pts = approx.reshape(4, 2).astype("float32")
        rect = _order_quad(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        max_w = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        max_h = int(max(heightA, heightB))
        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                       dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (max_w, max_h))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Deskew falló: %s — se mantiene la imagen original", exc)
        return img


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Reordena 4 puntos a TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ---------------------------------------------------------------------------
# OCR (opcional)
# ---------------------------------------------------------------------------

def _try_ocr(img: np.ndarray) -> Tuple[str, List[RegionOCR]]:
    """Corre Tesseract con --psm 11 si está disponible; si no, devuelve vacío."""
    try:
        import pytesseract  # type: ignore
    except ImportError:
        logger.warning("pytesseract no instalado — saltando OCR")
        return "", []
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 11")
        data = pytesseract.image_to_data(
            gray, config="--psm 11", output_type=pytesseract.Output.DICT
        )
        regions: List[RegionOCR] = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            regions.append(RegionOCR(
                text=txt.upper(),
                box=(int(data["left"][i]), int(data["top"][i]),
                     int(data["width"][i]), int(data["height"][i])),
                conf=conf,
            ))
        return text.upper(), regions
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tesseract falló (%s) — se devuelve OCR vacío", exc)
        return "", []


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

class PreprocessingModule:
    """Ejecuta deskew + resize + OCR sobre una imagen de boleta."""

    def __init__(self, target_long_edge: int = 1600,
                 backbone_input_size: int = 224) -> None:
        self.target_long_edge = target_long_edge
        self.backbone_input_size = backbone_input_size

    def process(self, image: Union[str, Path, np.ndarray, Image.Image],
                apply_geometry: bool = True) -> PreprocessOutput:
        """Procesa la imagen y devuelve los 4 artefactos.

        Acepta path, ndarray RGB o PIL.Image. Cuando ``apply_geometry=False``
        se evita el deskew/upscale: útil cuando ya conocemos el layout exacto
        (caso del demo sintético con ``boxes_hint``).
        """
        img = self._to_rgb_array(image)
        if apply_geometry:
            img = _try_deskew(img)
            img = _resize_long_edge(img, self.target_long_edge)

        texto, regiones = _try_ocr(img)

        backbone_pil = Image.fromarray(img).resize(
            (self.backbone_input_size, self.backbone_input_size),
            resample=Image.BILINEAR,
        )

        return PreprocessOutput(
            imagen_alta_res=img,
            imagen_backbone=backbone_pil,
            texto_ocr=texto,
            regiones_ocr=regiones,
        )

    @staticmethod
    def _to_rgb_array(image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if arr is None:
                raise FileNotFoundError(f"No se pudo cargar: {image}")
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image
        raise TypeError(f"Tipo de imagen no soportado: {type(image)}")
