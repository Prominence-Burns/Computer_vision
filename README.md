# Computer Vision — Clasificación de Boletas Electorales

Repositorio del proyecto de visión por computadora para análisis y clasificación automática de boletas electorales mexicanas.

## Descripción

Pipeline de Machine Learning que procesa fotografías de boletas electorales tomadas con dispositivos móviles, generando salida estructurada compatible con el Acta de Escrutinio y Cómputo de Casilla (AECC) del INE.

## Contenido

| Archivo | Descripción |
|---|---|
| `actas_escrutinio.pdf` | Referencia oficial de actas de escrutinio |
| `Cuadernillo-de-consulta-sobre-votos-validos-y-nulos-1.pdf` | Guía INE de votos válidos y nulos |
| `aecc_especificacion.md` | Especificación del formato AECC (salida JSON del pipeline) |
| `prompt_pipeline_ml_boletas.md` | Diseño del pipeline ML — arquitectura, datos y métricas |

## Fundamento Legal

- Arts. 288, 290, 291, 292 LGIPE
- Acuerdo INE/CG598/2023
- Proceso Electoral Federal 2023-2024

## Stack Tecnológico (planeado)

- Python · PyTorch / TensorFlow
- OpenCV · Detectron2 / YOLO
- Tesseract OCR
- FastAPI (serving)

## Organización

[Prominence Burns](https://github.com/prominence-burns)
