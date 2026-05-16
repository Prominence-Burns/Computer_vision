# EXP1 — MVP del pipeline ML de boletas

Implementación inicial del pipeline descrito en `../prompt_pipeline_ml_boletas.md`. Cubre:

- **M1** preprocesamiento + OCR
- **M2** detección de marcas
- **M5** reglas R1–R5 + cabeza neuronal mini (EfficientNet-B0 + Linear)
- **M9** JSON de boleta (`urn:ine:aecc:boleta:v1`)
- **M10** JSON agregado de casilla (`urn:ine:aecc:casilla:v1`)

Quedan fuera de esta entrega: M3 (header SIFT), 3 backbones de M4, capa difusa M5.7, adaptador a base de datos PREP.

## Instalación

```bash
cd EXP1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pytesseract` requiere el binario `tesseract` instalado en el sistema (`apt install tesseract-ocr`). Si no está disponible, el módulo M1 hace skip de OCR con un warning y el pipeline sigue funcionando (las reglas R1–R3 siguen activas).

## Demo end-to-end (sintético)

```bash
python -m pipeline_boletas.main demo
```

Genera ~200 boletas sintéticas con PIL, entrena la cabeza de clasificación 3 epochs en CPU, evalúa sobre el split de test y produce:

- `outputs/casilla_demo.json` — JSON de casilla agregado conforme a `urn:ine:aecc:casilla:v1`.
- `outputs/eval_report.json` — métricas (F1 macro/micro, exact destinatario accuracy, confusion matrix).
- `outputs/training_metrics.json` — curva de loss por epoch.
- `outputs/model_head.pt` — pesos de la cabeza entrenada.
- `outputs/boleta_demo.png` — una boleta sintética de ejemplo.

## Clasificar una imagen real

```bash
python -m pipeline_boletas.main classify \
    --imagen ruta/a/foto_boleta.jpg \
    --casilla NL-01-865-B \
    --tipo-boleta diputacion_mr
```

Produce un JSON conforme a `urn:ine:aecc:boleta:v1` en `outputs/`. Si no hay modelo entrenado en `outputs/model_head.pt`, el clasificador opera sólo con reglas (R1–R5) sobre las marcas detectadas por M2.

## Estructura

```
pipeline_boletas/
├── config.py        # CONFIG + enums + constantes AECC
├── preprocessing.py # M1
├── marks.py         # M2
├── features.py      # backbone EfficientNet-B0 congelado
├── classifier.py    # M5.5 cabeza + masking
├── rules.py         # M5.4 R1–R5
├── synthetic.py     # M6 generador
├── training.py      # M7 loop
├── evaluation.py    # M8 métricas
├── boleta_json.py   # M9
├── casilla_json.py  # M10
└── main.py          # CLI
```

## Imágenes seed

Coloca fotografías reales de boletas en `data/seed/` para que la augmentation las use como base adicional y el health-check sintético-vs-real reporte el dominio gap. Si está vacío, el demo opera 100% sintético.
