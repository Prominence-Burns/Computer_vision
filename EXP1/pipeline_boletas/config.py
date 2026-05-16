"""Configuración global, enums y constantes del pipeline AECC.

Todos los hiperparámetros editables viven en ``CONFIG``. Las constantes
electorales (PARTIDOS, COALICIONES, FUNDAMENTO_LEGAL) son contrato del
schema AECC y no decisiones de diseño: se copian literalmente del prompt
maestro.
"""
from __future__ import annotations

import os
import random
from enum import Enum
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Catálogo electoral 2024 (alineado con urn:ine:aecc:*:v1)
# ---------------------------------------------------------------------------

PARTIDOS: List[str] = [
    "PAN",
    "PRI",
    "PRD",
    "PVEM",
    "PT",
    "MC",
    "MORENA",
    "CI",
    "NULO",
]

CATALOGO_RECUADROS: List[str] = [
    "PAN", "PRI", "PRD", "PVEM", "PT", "MC", "MORENA", "CI", "SHH", "FCM", "CNR",
]

COALICIONES_2024: Dict[str, Dict] = {
    "FCM": {
        "nombre": "Fuerza y Corazón por México",
        "partidos": ["PAN", "PRI", "PRD"],
    },
    "SHH": {
        "nombre": "Sigamos Haciendo Historia",
        "partidos": ["PVEM", "PT", "MORENA"],
    },
}


class TipoBoleta(Enum):
    PRESIDENCIA = "presidencia"
    DIPUTACION_MR = "diputacion_mr"
    DIPUTACION_RP = "diputacion_rp"
    SENADO_MR = "senado_mr"
    SENADO_RP = "senado_rp"
    MUNICIPAL = "municipal"
    DESCONOCIDO = "DESCONOCIDO"


class TipoClasificacion(Enum):
    VALIDO = "valido"
    NULO = "nulo"
    NO_REGISTRADO = "no_registrado"


class SubtipoClasificacion(Enum):
    # Válidos
    MARCA_ESTANDAR = "marca_estandar"
    MARCA_ATIPICA = "marca_atipica"
    MARCA_FUERA_RECUADRO = "marca_fuera_recuadro"
    RECUADRO_ENCERRADO = "recuadro_encerrado"
    TEXTO_NO_OFENSIVO = "texto_no_ofensivo"
    MULTIMARCA_POSITIVA = "multimarca_positiva"
    NOMINATIVO_NOMBRE = "nominativo_nombre"
    NOMINATIVO_APODO = "nominativo_apodo"
    COALICION_MULTIMARCA = "coalicion_multimarca"
    MARCA_TENUE_PATRON = "marca_tenue_patron"
    # Nulos
    MARCA_TOTAL = "marca_total"
    INSULTO = "insulto"
    MULTIMARCA_NO_COALIGADOS = "multimarca_no_coaligados"
    BLANCO = "blanco"
    ROTURA_GRAVE = "rotura_grave"
    NOMINATIVO_CONTRADICTORIO = "nominativo_contradictorio"
    # No registrado
    NOMBRE_CANDIDATO_NR = "nombre_candidato_nr"
    SIGLAS_NR = "siglas_nr"


FUNDAMENTO_LEGAL: Dict[str, str] = {
    "marca_estandar": "Art. 288 LGIPE",
    "marca_atipica": "SUP-JIN-081/2006",
    "marca_fuera_recuadro": "SUP-JIN-021/2006",
    "recuadro_encerrado": "SUP-JIN-005/2006",
    "texto_no_ofensivo": "SUP-JIN-051/2012",
    "multimarca_positiva": "SUP-JIN-011/2012",
    "nominativo_nombre": "SUP-JIN-246/2006",
    "nominativo_apodo": "INE/CG517/2018",
    "coalicion_multimarca": "Art. 288 párrafo 3 LGIPE",
    "marca_tenue_patron": "SUP-JIN-014/2012",
    "marca_total": "SM-JIN-046/2015",
    "insulto": "SUP-JIN-069/2006",
    "multimarca_no_coaligados": "SUP-JIN-028/2012",
    "blanco": "Art. 291 pár.1 inc.b) LGIPE – SUP-JIN-081/2006",
    "rotura_grave": "SUP-JIN-085/2006",
    "nominativo_contradictorio": "INE/CG517/2018",
    "nombre_candidato_nr": "SUP-JIN-246/2006",
    "siglas_nr": "SM-JIN-046/2015",
}


CLASES_POR_BOLETA: Dict[TipoBoleta, List[str]] = {
    TipoBoleta.PRESIDENCIA: PARTIDOS,
    TipoBoleta.DIPUTACION_MR: PARTIDOS,
    TipoBoleta.DIPUTACION_RP: [p for p in PARTIDOS if p != "CI"],
    TipoBoleta.SENADO_MR: PARTIDOS,
    TipoBoleta.SENADO_RP: [p for p in PARTIDOS if p != "CI"],
    TipoBoleta.MUNICIPAL: PARTIDOS,
}


# Distribución de etiquetas para el dataset sintético (suma ≈ 1.0).
DISTRIBUCION_ETIQUETAS: Dict[str, float] = {
    "FCM": 0.15,
    "SHH": 0.30,
    "MC": 0.18,
    "PAN": 0.06,
    "PRI": 0.06,
    "PRD": 0.04,
    "PVEM": 0.04,
    "PT": 0.04,
    "MORENA": 0.04,
    "CI": 0.02,
    "NULO": 0.07,
}


# Listas de palabras simples para R4 (texto OCR). En producción esto
# debería reemplazarse por catálogos del INE (candidatos registrados,
# tesauro de insultos validado por el TEPJF).
PALABRAS_OFENSIVAS: List[str] = [
    "PENDEJO", "IDIOTA", "RATERO", "CORRUPTO", "ASESINO",
]
NOMBRES_CANDIDATOS_REGISTRADOS: List[str] = []  # placeholder
SIGLAS_NO_REGISTRADAS: List[str] = []           # placeholder


# ---------------------------------------------------------------------------
# CONFIG global
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG: Dict = {
    "seed": 42,

    # Generación sintética
    "synthetic_n_source_images": 40,    # imágenes "fuente" antes de augmentation
    "synthetic_n_augmentations": 5,     # augmentations por fuente → ~200 muestras
    "synthetic_image_size": (640, 880),  # (W, H) — boleta retrato

    # Backbone
    "backbone_id": "google/efficientnet-b0",
    "backbone_dim": 1280,
    "backbone_input_size": 224,

    # Cabeza de clasificación
    "head_hidden_dim": 256,
    "head_dropout": 0.4,
    "n_classes": len(PARTIDOS),         # 9
    "decision_threshold": 0.5,

    # Entrenamiento
    "num_epochs": 3,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,

    # Splits
    "split_train": 0.70,
    "split_val": 0.15,
    "split_test": 0.15,

    # Reglas R5 — umbrales de confianza
    "confianza_auto": 0.85,
    "confianza_revision": 0.60,

    # Texto / R4
    "palabras_ofensivas": PALABRAS_OFENSIVAS,
    "nombres_candidatos_registrados": NOMBRES_CANDIDATOS_REGISTRADOS,
    "siglas_no_registradas": SIGLAS_NO_REGISTRADAS,

    # Rutas
    "outputs_dir": REPO_ROOT / "outputs",
    "seed_images_dir": REPO_ROOT / "data" / "seed",
    "model_path": REPO_ROOT / "outputs" / "model_head.pt",

    # Eventos PREP (M9)
    "eventos": {
        "ballot_scanned": "siempre al procesar una boleta",
        "vote_detected": "si clasificacion.tipo != None",
        "inconsistency_detected": "si requiere_revision=True o anomalias no vacío",
    },

    # Pipeline metadata
    "pipeline_version": "0.1.0-EXP1",
}


CONFIANZA_AUTO: float = CONFIG["confianza_auto"]
CONFIANZA_REVISION: float = CONFIG["confianza_revision"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def coalition_to_onehot(destinatario_id: str):
    """Convierte un ID de destinatario al vector one-hot de 9 posiciones.

    Para coaliciones activa todos los partidos miembros (R1 se encarga del
    colapso inverso en inferencia). ``NULO`` activa el slot dedicado.
    """
    import torch

    onehot = torch.zeros(len(PARTIDOS))
    if destinatario_id in COALICIONES_2024:
        for partido in COALICIONES_2024[destinatario_id]["partidos"]:
            onehot[PARTIDOS.index(partido)] = 1.0
    elif destinatario_id in PARTIDOS:
        onehot[PARTIDOS.index(destinatario_id)] = 1.0
    elif destinatario_id == "NULO":
        onehot[PARTIDOS.index("NULO")] = 1.0
    else:
        raise ValueError(f"destinatario_id desconocido: {destinatario_id}")
    return onehot


def set_global_seed(seed: int = 42) -> None:
    """Fija las seeds de Python, NumPy y PyTorch para reproducibilidad."""
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass


def ensure_outputs_dir() -> Path:
    out = CONFIG["outputs_dir"]
    out.mkdir(parents=True, exist_ok=True)
    return out
