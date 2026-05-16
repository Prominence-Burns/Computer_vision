"""CLI: ``python -m pipeline_boletas.main {demo|classify}``.

Subcomandos:
- ``demo``: genera dataset sintético, entrena la cabeza 3 epochs, evalúa
  sobre el split de test y produce un JSON de casilla agregado.
- ``classify --imagen ruta.jpg --casilla NL-01-865-B``: corre el pipeline
  M1→M2→M5→M9 sobre una imagen real (o sintética).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from .boleta_json import build_boleta_json, dump_to_path as dump_boleta
from .casilla_json import CasillaAggregator, dump_to_path as dump_casilla
from .classifier import infer
from .config import (
    CONFIG,
    PARTIDOS,
    TipoBoleta,
    ensure_outputs_dir,
    set_global_seed,
)
from .evaluation import (
    confusion_subtipos,
    evaluate_classifier,
    evaluate_destinatarios,
    health_check_sintetico,
    tasa_requiere_revision,
    write_evaluation_report,
)
from .features import EfficientNetExtractor
from .marks import MarkDetectionModule
from .preprocessing import PreprocessingModule
from .rules import aplicar_R1_R5
from .synthetic import SyntheticBallotGenerator, render_single_demo_ballot
from .training import load_trained_head, train_head


logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline_boletas")


# ---------------------------------------------------------------------------
# Pipeline de inferencia para una sola boleta
# ---------------------------------------------------------------------------

def classify_single(image,
                    *,
                    casilla_id: str,
                    tipo_boleta: TipoBoleta,
                    extractor: Optional[EfficientNetExtractor],
                    head,
                    boxes_hint=None,
                    image_url: Optional[str] = None,
                    boleta_id: Optional[str] = None,
                    timestamp: Optional[str] = None) -> Dict[str, Any]:
    """Devuelve el JSON AECC de una boleta."""
    pre_module = PreprocessingModule(
        target_long_edge=1600,
        backbone_input_size=CONFIG["backbone_input_size"],
    )
    mark_module = MarkDetectionModule()

    pre_out = pre_module.process(image, apply_geometry=(boxes_hint is None))
    marcas = mark_module.detect(pre_out.imagen_alta_res, boxes_hint=boxes_hint)

    probs = None
    if extractor is not None and head is not None:
        feats = extractor.extract([pre_out.imagen_backbone])
        _, probs_t = infer(head, feats, tipo_boleta)
        probs = probs_t[0]

    resultado = aplicar_R1_R5(marcas, probs, pre_out.texto_ocr, tipo_boleta)
    return build_boleta_json(
        resultado=resultado,
        marcas=marcas,
        casilla_id=casilla_id,
        tipo_boleta=tipo_boleta.value,
        boleta_id=boleta_id,
        image_url=image_url,
        texto_detectado=pre_out.texto_ocr or None,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Subcomando: demo
# ---------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> None:
    set_global_seed(CONFIG["seed"])
    out_dir = ensure_outputs_dir()

    # 1) Dataset sintético
    logger.info("Generando dataset sintético…")
    gen = SyntheticBallotGenerator(seed=CONFIG["seed"])
    samples = gen.build_dataset(
        n_sources=args.n_sources or CONFIG["synthetic_n_source_images"],
        n_aug_per_source=args.n_aug if args.n_aug is not None else CONFIG["synthetic_n_augmentations"],
        tipo_boleta=TipoBoleta.DIPUTACION_MR,
    )
    logger.info("Generadas %d muestras (%d fuentes × %d augmentations)",
                len(samples), args.n_sources or CONFIG["synthetic_n_source_images"],
                1 + (args.n_aug if args.n_aug is not None else CONFIG["synthetic_n_augmentations"]))
    # Guardar 1 boleta de muestra para inspección visual
    samples[0].image.save(out_dir / "boleta_demo.png")

    # 2) Backbone + entrenamiento
    extractor = EfficientNetExtractor()
    train_result = train_head(samples, extractor, num_epochs=args.epochs or CONFIG["num_epochs"])

    # 3) Evaluación neuronal pura
    head = train_result.model
    head.eval()
    with torch.inference_mode():
        test_logits = head(train_result.test_features)
        test_scores = torch.sigmoid(test_logits)
        test_pred = (test_scores >= CONFIG["decision_threshold"]).int()
    classifier_metrics = evaluate_classifier(
        train_result.test_labels, test_pred, y_scores=test_scores
    )

    # 4) Pipeline end-to-end sobre el split de test:
    #    re-aplicar R1–R5 + construir JSON de boleta para cada muestra
    test_samples = [samples[i] for i in train_result.test_indices]
    boletas_jsons: List[Dict[str, Any]] = []
    destinatarios_true: List[str | None] = []
    destinatarios_pred: List[str | None] = []
    subtipos_true: List[str] = []
    subtipos_pred: List[str] = []
    requieren_revision: List[bool] = []
    casilla_id = "DEMO-001-001-B"

    for i, sample in enumerate(test_samples):
        boleta = classify_single(
            image=sample.image,
            casilla_id=casilla_id,
            tipo_boleta=sample.tipo_boleta,
            extractor=extractor,
            head=head,
            boxes_hint=sample.boxes,
            image_url=f"synthetic://{sample.source_image_id}",
            boleta_id=f"B-DEMO-{i:04d}",
            timestamp="2024-06-02T18:00:00.000Z",   # determinístico para hash reproducible
        )
        boletas_jsons.append(boleta)
        destinatarios_true.append(sample.label_id if sample.label_id != "NULO" else None)
        destinatarios_pred.append(boleta["clasificacion"]["destinatario"])
        # ground-truth subtipo aproximado
        if sample.label_id == "NULO":
            subtipos_true.append("blanco")
        elif sample.label_id in ("SHH", "FCM"):
            subtipos_true.append("coalicion_multimarca")
        else:
            subtipos_true.append("marca_estandar")
        subtipos_pred.append(boleta["clasificacion"]["subtipo"])
        requieren_revision.append(boleta["clasificacion"]["requiere_revision"])

    dest_metrics = evaluate_destinatarios(destinatarios_true, destinatarios_pred)
    matrix = confusion_subtipos(subtipos_true, subtipos_pred)
    revision_rate = tasa_requiere_revision(requieren_revision)

    # 5) JSON de casilla agregado
    metadatos = {
        "casilla_id": casilla_id,
        "entidad_federativa": "Demo",
        "municipio_o_delegacion": "Demo",
        "distrito": "01",
        "seccion": "0001",
        "tipo_casilla": "basica",
        "tipo_eleccion": TipoBoleta.DIPUTACION_MR.value,
        "proceso_electoral": "2023-2024",
        "boletas_recibidas": len(boletas_jsons) + 50,
        "BS": 50,
        "PV": len(boletas_jsons),
        "RPPV": 0,
    }
    aggregator = CasillaAggregator()
    casilla = aggregator.aggregate(boletas_jsons, metadatos)
    dump_casilla(casilla, out_dir / "casilla_demo.json")

    # 6) Reporte de evaluación
    report = {
        "primary": {
            "exact_destinatario_accuracy": dest_metrics["exact_destinatario_accuracy"],
            "f1_macro": classifier_metrics["f1_macro"],
            "f1_micro": classifier_metrics["f1_micro"],
            "tasa_requiere_revision": revision_rate,
            "confusion_subtipos": matrix,
        },
        "secondary": {
            "hamming_loss": classifier_metrics["hamming_loss"],
            "subset_accuracy": classifier_metrics["subset_accuracy"],
            "lrap": classifier_metrics.get("lrap"),
            "coverage_error": classifier_metrics.get("coverage_error"),
        },
        "n_test": len(test_samples),
        "hash_boletas": casilla["hash_boletas"],
        "consistencia_acta": casilla["consistencia"]["acta_consistente"],
    }

    # 7) Health check sintético-vs-real (si hay imágenes en data/seed/)
    seed_dir = Path(CONFIG["seed_images_dir"])
    seed_imgs = sorted([p for p in seed_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if seed_imgs:
        emb_real = extractor.extract_many([Image.open(p).convert("RGB") for p in seed_imgs])
        report["domain_gap_synthetic_vs_real"] = health_check_sintetico(
            train_result.test_features, emb_real
        )
    write_evaluation_report(report)

    # Resumen consola
    logger.info("=== RESUMEN DEL DEMO ===")
    logger.info("F1 macro (cabeza)                 : %.3f", classifier_metrics["f1_macro"])
    logger.info("F1 micro (cabeza)                 : %.3f", classifier_metrics["f1_micro"])
    logger.info("Exact destinatario accuracy        : %.3f", dest_metrics["exact_destinatario_accuracy"])
    logger.info("Tasa requiere_revision             : %.3f", revision_rate)
    logger.info("Hash determinista de boletas        : %s", casilla["hash_boletas"])
    logger.info("Acta consistente                    : %s", casilla["consistencia"]["acta_consistente"])
    logger.info("Salidas en %s", out_dir)


# ---------------------------------------------------------------------------
# Subcomando: classify (una sola imagen)
# ---------------------------------------------------------------------------

def cmd_classify(args: argparse.Namespace) -> None:
    set_global_seed(CONFIG["seed"])
    out_dir = ensure_outputs_dir()

    tipo_boleta = TipoBoleta(args.tipo_boleta)
    image_path = Path(args.imagen)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    head = None
    extractor = None
    if not args.solo_reglas:
        try:
            extractor = EfficientNetExtractor()
            head = load_trained_head(CONFIG["model_path"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo cargar el clasificador (%s) — modo solo reglas", exc)
            extractor, head = None, None

    boleta = classify_single(
        image=str(image_path),
        casilla_id=args.casilla,
        tipo_boleta=tipo_boleta,
        extractor=extractor,
        head=head,
        image_url=str(image_path),
        boleta_id=args.boleta_id,
    )
    out_path = out_dir / f"boleta_{boleta['boleta_id']}.json"
    dump_boleta(boleta, out_path)
    print(json.dumps(boleta, indent=2, ensure_ascii=False))
    logger.info("JSON de boleta escrito en %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline_boletas",
                                      description="MVP del pipeline ML AECC.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="Demo end-to-end con dataset sintético.")
    p_demo.add_argument("--n-sources", type=int, default=None,
                         help=f"Imágenes fuente antes de augmentation (default {CONFIG['synthetic_n_source_images']})")
    p_demo.add_argument("--n-aug", type=int, default=None,
                         help=f"Augmentations por fuente (default {CONFIG['synthetic_n_augmentations']})")
    p_demo.add_argument("--epochs", type=int, default=None,
                         help=f"Epochs de entrenamiento (default {CONFIG['num_epochs']})")
    p_demo.set_defaults(func=cmd_demo)

    p_clf = sub.add_parser("classify", help="Clasifica una imagen → JSON AECC de boleta.")
    p_clf.add_argument("--imagen", required=True, help="Ruta a la fotografía de boleta.")
    p_clf.add_argument("--casilla", required=True, help="ID de casilla.")
    p_clf.add_argument("--tipo-boleta", default="diputacion_mr",
                        choices=[t.value for t in TipoBoleta if t != TipoBoleta.DESCONOCIDO])
    p_clf.add_argument("--boleta-id", default=None)
    p_clf.add_argument("--solo-reglas", action="store_true",
                        help="No carga el backbone — sólo M1+M2+R1–R5.")
    p_clf.set_defaults(func=cmd_classify)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
