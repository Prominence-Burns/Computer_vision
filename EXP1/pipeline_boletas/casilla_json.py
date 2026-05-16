"""M10 — Agregador AECC de casilla (urn:ine:aecc:casilla:v1).

Mapeo a entidades PREP:
- JSON casilla completo → ``tally_sheets``
- ``bloque_2.resultados[]`` → ``results`` (1 fila por partido/coalición)
- Evento ``results_submitted`` → ``events``
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import COALICIONES_2024, PARTIDOS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hash canónico (SHA-256 reproducible)
# ---------------------------------------------------------------------------

def hash_boletas(boletas: List[Dict[str, Any]]) -> str:
    """SHA-256 sobre la serialización canónica del array de boletas."""
    boletas_ordenadas = sorted(boletas, key=lambda b: b["boleta_id"])
    canon = json.dumps(
        boletas_ordenadas,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


# ---------------------------------------------------------------------------
# Helpers de agregación
# ---------------------------------------------------------------------------

_NOMBRE_HUMANO = {
    "PAN": "Partido Acción Nacional",
    "PRI": "Partido Revolucionario Institucional",
    "PRD": "Partido de la Revolución Democrática",
    "PVEM": "Partido Verde Ecologista de México",
    "PT": "Partido del Trabajo",
    "MC": "Movimiento Ciudadano",
    "MORENA": "Morena",
    "CI": "Candidatura Independiente",
}


def _resultados_por_destinatario(boletas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conteo: Dict[str, int] = {}
    for b in boletas:
        if b["clasificacion"]["tipo"] != "valido":
            continue
        dest = b["clasificacion"]["destinatario"]
        if dest is None or dest == "CNR":
            continue
        conteo[dest] = conteo.get(dest, 0) + 1

    resultados: List[Dict[str, Any]] = []
    for dest_id, votos in sorted(conteo.items(), key=lambda kv: -kv[1]):
        if dest_id in COALICIONES_2024:
            resultados.append({
                "partido_o_coalicion": COALICIONES_2024[dest_id]["nombre"],
                "id": dest_id,
                "es_coalicion": True,
                "partidos_coalicion": list(COALICIONES_2024[dest_id]["partidos"]),
                "votos": votos,
            })
        else:
            resultados.append({
                "partido_o_coalicion": _NOMBRE_HUMANO.get(dest_id, dest_id),
                "id": dest_id,
                "es_coalicion": False,
                "partidos_coalicion": None,
                "votos": votos,
            })
    return resultados


def _consistencia(metadatos: Dict[str, Any], bsu: int, votos_validos: int,
                  cnr: int, vn: int) -> Dict[str, Any]:
    pv = int(metadatos.get("PV", 0))
    rppv = int(metadatos.get("RPPV", 0))
    sv = pv + rppv
    rv = votos_validos + cnr + vn

    crit_1 = (pv + rppv) == sv
    crit_2 = sv == bsu
    crit_3 = bsu == rv
    crit_4 = (votos_validos + cnr + vn) == rv
    consistente = all([crit_1, crit_2, crit_3, crit_4])

    if consistente:
        tipo_error = "ninguno"
    else:
        errores = []
        if not crit_1: errores.append("pv_rppv_sv")
        if not crit_2: errores.append("sv_bsu")
        if not crit_3: errores.append("bsu_rv")
        if not crit_4: errores.append("sum_vi_rv")
        tipo_error = ",".join(errores)

    return {
        "criterio_1_pv_rppv_sv": crit_1,
        "criterio_2_sv_bsu": crit_2,
        "criterio_3_bsu_rv": crit_3,
        "criterio_4_sum_vi_rv": crit_4,
        "acta_consistente": consistente,
        "tipo_error": tipo_error,
        "_valores": {"PV": pv, "RPPV": rppv, "SV": sv, "BSU": bsu, "RV": rv},
    }


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

class CasillaAggregator:
    """Construye el JSON AECC a partir de las boletas clasificadas."""

    def aggregate(self,
                  boletas: List[Dict[str, Any]],
                  metadatos: Dict[str, Any],
                  incidentes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not boletas:
            logger.warning("Lista de boletas vacía: el JSON de casilla quedará en ceros.")

        votos_validos = sum(1 for b in boletas
                            if b["clasificacion"]["tipo"] == "valido"
                            and b["clasificacion"]["destinatario"] not in (None, "CNR"))
        cnr = sum(1 for b in boletas if b["clasificacion"]["tipo"] == "no_registrado")
        vn = sum(1 for b in boletas if b["clasificacion"]["tipo"] == "nulo")

        bsu = len(boletas)
        rv = votos_validos + cnr + vn

        bloque_1 = {
            "boletas_recibidas": int(metadatos.get("boletas_recibidas", 0)),
            "BS": int(metadatos.get("BS", 0)),
            "PV": int(metadatos.get("PV", 0)),
            "RPPV": int(metadatos.get("RPPV", 0)),
            "SV": int(metadatos.get("PV", 0)) + int(metadatos.get("RPPV", 0)),
            "BSU": bsu,
        }
        bloque_2 = {
            "resultados": _resultados_por_destinatario(boletas),
            "CNR": cnr,
            "VN": vn,
            "RV": rv,
        }
        consistencia = _consistencia(metadatos, bsu, votos_validos, cnr, vn)

        boletas_revision = sum(1 for b in boletas
                                if b["clasificacion"]["requiere_revision"])

        return {
            "$schema": "urn:ine:aecc:casilla:v1",
            "metadatos": {k: v for k, v in metadatos.items()
                           if k in {"casilla_id", "entidad_federativa",
                                    "municipio_o_delegacion", "distrito",
                                    "seccion", "tipo_casilla", "tipo_eleccion",
                                    "proceso_electoral"}},
            "bloque_1": bloque_1,
            "bloque_2": bloque_2,
            "consistencia": consistencia,
            "incidentes": incidentes or {
                "se_presentaron": False,
                "descripcion": None,
                "hojas_de_incidentes": 0,
            },
            "boletas_procesadas": bsu,
            "boletas_revision_humana": boletas_revision,
            "hash_boletas": hash_boletas(boletas),
        }


def dump_to_path(casilla: Dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(casilla, f, indent=2, ensure_ascii=False)
    return path
