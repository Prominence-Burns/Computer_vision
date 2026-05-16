"""M5.4 — Reglas de post-procesamiento AECC R1–R5.

Capa lógica (NO neuronal) que toma:
- ``marcas`` (list[MarcaDetectada] del Módulo 2),
- ``probs`` (probabilidades sigmoideas del clasificador),
- ``texto_ocr`` (string limpio del Módulo 1),
- ``tipo_boleta`` (enum del Módulo 3 / parámetro),

y produce el ``ResultadoClasificacion`` final que se vuelca al JSON AECC.

Precedencia (per la nota arquitectónica del prompt):
    Módulo 2 (marcas) + R1–R5 → resultado primario
    Cabeza neuronal           → soporte para casos ambiguos / confianza
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import torch

from .config import (
    CONFIANZA_AUTO,
    CONFIANZA_REVISION,
    CONFIG,
    COALICIONES_2024,
    FUNDAMENTO_LEGAL,
    PARTIDOS,
    SubtipoClasificacion,
    TipoBoleta,
    TipoClasificacion,
)
from .marks import MarcaDetectada


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class ResultadoClasificacion:
    destinatario: Optional[str]
    tipo: TipoClasificacion
    subtipo: SubtipoClasificacion
    confianza: float
    requiere_revision: bool
    fundamento_legal: str
    one_hot_interno: List[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _es_marca_positiva(m: MarcaDetectada) -> bool:
    """Una marca cuenta como voto positivo si tiene tipo ✓ y está dentro."""
    return (
        m.tipo_marca in {"cruz", "raya", "palomita", "circulo"}
        and m.intensidad in {"clara", "tenue"}
        and m.dentro_recuadro
        and m.proporcion_en_recuadro >= 0.5
    )


def _coalicion_de(recuadros_marcados: Set[str]) -> Optional[str]:
    """Devuelve el ID de coalición si las marcas son ⊆ alguna coalición."""
    for coal_id, coal in COALICIONES_2024.items():
        miembros = set(coal["partidos"])
        if recuadros_marcados and recuadros_marcados.issubset(miembros) and len(recuadros_marcados) >= 2:
            return coal_id
    return None


def _palabras(texto: str) -> List[str]:
    return [w for w in (texto or "").upper().split() if w.isalpha()]


# ---------------------------------------------------------------------------
# Reglas R1–R5
# ---------------------------------------------------------------------------

def aplicar_R1_R5(marcas: Sequence[MarcaDetectada],
                  probs: Optional[torch.Tensor],
                  texto_ocr: str,
                  tipo_boleta: TipoBoleta) -> ResultadoClasificacion:
    """Determina destinatario, tipo, subtipo y requiere_revision.

    ``probs`` opcional: si viene None, la confianza se deriva de la
    intensidad media de las marcas (proxy) en lugar del clasificador.
    """
    marcas_positivas = [m for m in marcas if _es_marca_positiva(m)]
    recuadros_marcados = {m.recuadro for m in marcas_positivas}

    # ---------------- R4: análisis de texto OCR (alta prioridad) -----------
    palabras = _palabras(texto_ocr)
    palabras_ofensivas = set(CONFIG["palabras_ofensivas"])
    if palabras and any(w in palabras_ofensivas for w in palabras):
        return _resultado(
            destinatario=None,
            tipo=TipoClasificacion.NULO,
            subtipo=SubtipoClasificacion.INSULTO,
            confianza=0.95,
            requiere_revision=False,
            one_hot=_one_hot_from_recuadros(set()),
        )

    nombres_reg = set(CONFIG["nombres_candidatos_registrados"])
    if nombres_reg and any(w in nombres_reg for w in palabras) and not marcas_positivas:
        return _resultado(
            destinatario=None,
            tipo=TipoClasificacion.VALIDO,
            subtipo=SubtipoClasificacion.NOMINATIVO_NOMBRE,
            confianza=0.80,
            requiere_revision=True,
            one_hot=_one_hot_from_recuadros(set()),
        )

    siglas_nr = set(CONFIG["siglas_no_registradas"])
    if siglas_nr and any(w in siglas_nr for w in palabras) and not marcas_positivas:
        return _resultado(
            destinatario="CNR",
            tipo=TipoClasificacion.NO_REGISTRADO,
            subtipo=SubtipoClasificacion.SIGLAS_NR,
            confianza=0.85,
            requiere_revision=False,
            one_hot=_one_hot_from_recuadros(set()),
        )

    # ---------------- R1: coalición ----------------------------------------
    coalicion_id = _coalicion_de(recuadros_marcados)
    if coalicion_id is not None:
        confianza = _confianza_desde_marcas(marcas_positivas, probs)
        return _resultado(
            destinatario=coalicion_id,
            tipo=TipoClasificacion.VALIDO,
            subtipo=SubtipoClasificacion.COALICION_MULTIMARCA,
            confianza=confianza,
            requiere_revision=confianza < CONFIANZA_AUTO,
            one_hot=_one_hot_from_recuadros(recuadros_marcados),
        )

    # ---------------- Sin marcas → blanco ----------------------------------
    if not marcas:
        return _resultado(
            destinatario=None,
            tipo=TipoClasificacion.NULO,
            subtipo=SubtipoClasificacion.BLANCO,
            confianza=0.99,
            requiere_revision=False,
            one_hot=_one_hot_from_recuadros(set()),
        )

    # ---------------- Multimarca no-coaligados → nulo ----------------------
    partidos_marcados = {r for r in recuadros_marcados if r in PARTIDOS}
    if len(partidos_marcados) >= 2:
        return _resultado(
            destinatario=None,
            tipo=TipoClasificacion.NULO,
            subtipo=SubtipoClasificacion.MULTIMARCA_NO_COALIGADOS,
            confianza=0.90,
            requiere_revision=False,
            one_hot=_one_hot_from_recuadros(partidos_marcados),
        )

    # ---------------- R2: marca fuera de recuadro --------------------------
    if not marcas_positivas:
        # Hay marcas pero ninguna positiva → revisar la mejor proporción
        mejor = max(marcas, key=lambda m: m.proporcion_en_recuadro)
        if mejor.proporcion_en_recuadro >= 0.5 and not mejor.dentro_recuadro:
            confianza = _confianza_desde_marcas([mejor], probs)
            return _resultado(
                destinatario=mejor.recuadro if mejor.recuadro in PARTIDOS else None,
                tipo=TipoClasificacion.VALIDO,
                subtipo=SubtipoClasificacion.MARCA_FUERA_RECUADRO,
                confianza=confianza,
                requiere_revision=confianza < CONFIANZA_AUTO,
                one_hot=_one_hot_from_recuadros({mejor.recuadro}),
            )
        # Marca débil → revisión humana
        return _resultado(
            destinatario=None,
            tipo=TipoClasificacion.NULO,
            subtipo=SubtipoClasificacion.BLANCO,
            confianza=0.50,
            requiere_revision=True,
            one_hot=_one_hot_from_recuadros(set()),
        )

    # ---------------- R3: una sola marca positiva → voto válido estándar ---
    if len(partidos_marcados) == 1:
        recuadro = next(iter(partidos_marcados))
        marca = next(m for m in marcas_positivas if m.recuadro == recuadro)
        confianza = _confianza_desde_marcas([marca], probs)

        # Subtipo según tipo de marca: cruz/raya = estándar; palomita/circulo = atípica
        if marca.tipo_marca == "cruz":
            subtipo = SubtipoClasificacion.MARCA_ESTANDAR
        elif marca.tipo_marca == "circulo":
            subtipo = SubtipoClasificacion.RECUADRO_ENCERRADO
        else:
            subtipo = SubtipoClasificacion.MARCA_ATIPICA

        # Otras marcas en otros recuadros (ej. tachones) → multimarca positiva
        if len(marcas) > 1:
            subtipo = SubtipoClasificacion.MULTIMARCA_POSITIVA

        return _resultado(
            destinatario=recuadro,
            tipo=TipoClasificacion.VALIDO,
            subtipo=subtipo,
            confianza=confianza,
            requiere_revision=confianza < CONFIANZA_AUTO,
            one_hot=_one_hot_from_recuadros({recuadro}),
        )

    # ---------------- Marcas en SHH/FCM/CI/CNR sin partido --------------------
    no_partido = recuadros_marcados - set(PARTIDOS)
    if no_partido:
        recuadro = next(iter(no_partido))
        if recuadro in {"SHH", "FCM"}:
            return _resultado(
                destinatario=recuadro,
                tipo=TipoClasificacion.VALIDO,
                subtipo=SubtipoClasificacion.COALICION_MULTIMARCA,
                confianza=0.85,
                requiere_revision=False,
                one_hot=_one_hot_from_recuadros(set(COALICIONES_2024[recuadro]["partidos"])),
            )
        if recuadro == "CI":
            return _resultado(
                destinatario="CI",
                tipo=TipoClasificacion.VALIDO,
                subtipo=SubtipoClasificacion.MARCA_ESTANDAR,
                confianza=0.85,
                requiere_revision=False,
                one_hot=_one_hot_from_recuadros({"CI"}),
            )
        if recuadro == "CNR":
            return _resultado(
                destinatario="CNR",
                tipo=TipoClasificacion.NO_REGISTRADO,
                subtipo=SubtipoClasificacion.NOMBRE_CANDIDATO_NR,
                confianza=0.85,
                requiere_revision=True,
                one_hot=_one_hot_from_recuadros(set()),
            )

    # ---------------- Default: requiere revisión humana --------------------
    return _resultado(
        destinatario=None,
        tipo=TipoClasificacion.NULO,
        subtipo=SubtipoClasificacion.BLANCO,
        confianza=0.40,
        requiere_revision=True,
        one_hot=_one_hot_from_recuadros(set()),
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _confianza_desde_marcas(marcas: Iterable[MarcaDetectada],
                            probs: Optional[torch.Tensor]) -> float:
    """Combina la intensidad de las marcas con la prob máxima del clasificador."""
    marcas = list(marcas)
    base = 0.5
    if marcas:
        intens = [_intensidad_a_score(m.intensidad) for m in marcas]
        base = sum(intens) / len(intens)
    if probs is not None:
        # toma la probabilidad máxima entre las clases activas
        try:
            p = float(probs.max().item())
            return float(min(1.0, max(base, 0.5 * base + 0.5 * p)))
        except Exception:  # noqa: BLE001
            return base
    return base


def _intensidad_a_score(intensidad: str) -> float:
    return {"clara": 0.95, "tenue": 0.70, "borrosa": 0.45}.get(intensidad, 0.50)


def _one_hot_from_recuadros(recuadros: Set[str]) -> List[int]:
    """Vector one-hot de 9 posiciones (orden ``PARTIDOS``)."""
    vec = [0] * len(PARTIDOS)
    for r in recuadros:
        if r in PARTIDOS:
            vec[PARTIDOS.index(r)] = 1
    return vec


def _resultado(*,
               destinatario: Optional[str],
               tipo: TipoClasificacion,
               subtipo: SubtipoClasificacion,
               confianza: float,
               requiere_revision: bool,
               one_hot: List[int]) -> ResultadoClasificacion:
    # R5: ajusta requiere_revision al umbral si la regla específica no lo forzó
    if confianza < CONFIANZA_REVISION:
        requiere_revision = True
    elif confianza < CONFIANZA_AUTO and not requiere_revision:
        requiere_revision = True

    fundamento = FUNDAMENTO_LEGAL.get(subtipo.value, "")
    return ResultadoClasificacion(
        destinatario=destinatario,
        tipo=tipo,
        subtipo=subtipo,
        confianza=round(float(confianza), 4),
        requiere_revision=bool(requiere_revision),
        fundamento_legal=fundamento,
        one_hot_interno=one_hot,
    )
