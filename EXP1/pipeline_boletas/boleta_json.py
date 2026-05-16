"""M9 — Constructor del JSON AECC de boleta (urn:ine:aecc:boleta:v1).

Mapeo a la entidad ``ballots`` (tabla del PREP):
| Campo JSON                          | Columna DB              |
|-------------------------------------|-------------------------|
| boleta_id                           | id                      |
| casilla_id                          | polling_station_id      |
| image_url                           | image_url               |
| clasificacion.destinatario          | detected_vote           |
| clasificacion.confianza             | confidence_score        |
| clasificacion.requiere_revision     | reviewed_by_human       |
| clasificacion.tipo + subtipo        | final_classification    |
| timestamp                           | created_at              |

Eventos esperados (entidad ``events`` del PREP):
- ``ballot_scanned`` — siempre.
- ``vote_detected`` — si ``clasificacion.tipo`` no es ``None``.
- ``inconsistency_detected`` — si ``requiere_revision`` o ``anomalias`` no vacío.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import CONFIG, PARTIDOS
from .marks import MarcaDetectada
from .rules import ResultadoClasificacion


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def build_boleta_json(*,
                      resultado: ResultadoClasificacion,
                      marcas: Sequence[MarcaDetectada],
                      casilla_id: str,
                      tipo_boleta: str,
                      boleta_id: Optional[str] = None,
                      image_url: Optional[str] = None,
                      texto_detectado: Optional[str] = None,
                      anomalias: Optional[List[Dict[str, Any]]] = None,
                      timestamp: Optional[str] = None,
                      ) -> Dict[str, Any]:
    """Devuelve el dict JSON conforme a urn:ine:aecc:boleta:v1."""
    one_hot_dict = {p: int(v) for p, v in zip(PARTIDOS, resultado.one_hot_interno)}
    return {
        "$schema": "urn:ine:aecc:boleta:v1",
        "boleta_id": boleta_id or f"B-{uuid.uuid4().hex[:8].upper()}",
        "casilla_id": casilla_id,
        "timestamp": timestamp or _iso_now(),
        "image_url": image_url,
        "clasificacion": {
            "tipo": resultado.tipo.value,
            "subtipo": resultado.subtipo.value,
            "destinatario": resultado.destinatario,
            "confianza": resultado.confianza,
            "requiere_revision": resultado.requiere_revision,
        },
        "marcas_detectadas": [
            {
                "recuadro": m.recuadro,
                "tipo_marca": m.tipo_marca,
                "intensidad": m.intensidad,
                "dentro_recuadro": m.dentro_recuadro,
                "proporcion_en_recuadro": round(m.proporcion_en_recuadro, 4),
            }
            for m in marcas
        ],
        "anomalias": anomalias or [],
        "texto_detectado": texto_detectado or None,
        "fundamento_legal": resultado.fundamento_legal,
        "metadata": {
            "backbone_used": [CONFIG["backbone_id"]],
            "pipeline_version": CONFIG["pipeline_version"],
            "tipo_boleta": tipo_boleta,
            "one_hot_interno": one_hot_dict,
        },
    }


def emit_eventos(boleta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Genera los eventos PREP correspondientes a esta boleta."""
    eventos = [{
        "tipo": "ballot_scanned",
        "boleta_id": boleta["boleta_id"],
        "casilla_id": boleta["casilla_id"],
        "timestamp": boleta["timestamp"],
    }]
    if boleta["clasificacion"]["tipo"] is not None:
        eventos.append({
            "tipo": "vote_detected",
            "boleta_id": boleta["boleta_id"],
            "destinatario": boleta["clasificacion"]["destinatario"],
            "tipo_clasificacion": boleta["clasificacion"]["tipo"],
            "timestamp": boleta["timestamp"],
        })
    if boleta["clasificacion"]["requiere_revision"] or boleta["anomalias"]:
        eventos.append({
            "tipo": "inconsistency_detected",
            "boleta_id": boleta["boleta_id"],
            "motivo": "requiere_revision" if boleta["clasificacion"]["requiere_revision"]
                       else "anomalia",
            "timestamp": boleta["timestamp"],
        })
    return eventos


def dump_to_path(boleta: Dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(boleta, f, indent=2, ensure_ascii=False)
    return path
