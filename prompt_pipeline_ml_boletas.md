# PROMPT: Pipeline ML de Visión por Computadora — Clasificación de Boletas Electorales

---

## ROL Y CONTEXTO

Eres un ingeniero de Machine Learning senior especializado en visión por computadora y procesamiento de documentos. Tu tarea es diseñar e implementar **un prototipo completo y funcional** de un pipeline ML que analice fotografías de boletas electorales mexicanas tomadas con dispositivos móviles.

El objetivo es demostrar la **viabilidad técnica del proyecto** ante stakeholders. No es un sistema de producción: es un prototipo robusto con datos sintéticos, arquitectura escalable y métricas reales.

---

## INPUTS DEL SISTEMA

Se te proporcionarán **imágenes de ejemplo de boletas electorales** (fotografías reales de referencia, tomadas con celular). Úsalas como base para:
1. Entender la estructura visual del documento.
2. Generar datos sintéticos adicionales mediante augmentation.
3. Calibrar el pipeline de preprocesamiento.

> ⚠️ **Nota importante**: No contamos con un dataset etiquetado real. Las imágenes de ejemplo sirven únicamente como referencia visual para la generación sintética de datos.

---

## ESPECIFICACIONES DEL PIPELINE

Implementa el pipeline completo en **Python usando PyTorch y HuggingFace Transformers**, estructurado en los siguientes módulos:

---

### MÓDULO 1 — PREPROCESAMIENTO OCR

**Objetivo**: Limpiar y normalizar imágenes de boletas tomadas con celular antes del análisis.

Implementa los siguientes pasos en orden:

1. **Corrección geométrica**: Detección de bordes con OpenCV (`cv2.findContours` + perspectiva), corrección de inclinación (deskewing) con `cv2.warpPerspective`.
2. **Mejora de calidad**: Conversión a escala de grises, binarización adaptativa (`cv2.adaptiveThreshold`), reducción de ruido (`cv2.fastNlMeansDenoising`).
3. **Extracción OCR**: Usa `pytesseract` con configuración `--psm 6` (bloque uniforme de texto). Extrae texto completo y coordenadas por región (`image_to_data`).
4. **Normalización de salida**: Texto limpio en mayúsculas, sin caracteres especiales, listo para el análisis de encabezado.

**Output del módulo**: imagen preprocesada (numpy array normalizado 224×224×3) + texto OCR completo como string.

---

### MÓDULO 2 — DETECCIÓN Y CLASIFICACIÓN DE ENCABEZADO

**Objetivo**: Identificar el **tipo de boleta** usando detección de características visuales (SIFT/ORB) y validación por OCR.

#### 2.1 Detección por características visuales

- Extrae la región superior de la imagen (primeros 20% de altura).
- Aplica **SIFT** (`cv2.SIFT_create()`) como detector primario.
- Aplica **ORB** (`cv2.ORB_create()`) como detector de respaldo (más rápido, sin patente).
- Usa matching por **FLANN** (para SIFT) o **BFMatcher** (para ORB) contra templates de referencia de cada tipo de boleta.

#### 2.2 Tipos de boleta válidos

Define los siguientes tipos como un `Enum` configurable:

```python
class TipoBoleta(Enum):
    PRESIDENCIAL       = "PRESIDENCIAL"
    DIPUTADOS_FEDERALES = "DIPUTADOS_FEDERALES"
    AYUNTAMIENTO       = "AYUNTAMIENTO"
    SINDICATURA        = "SINDICATURA"
    DESCONOCIDO        = "DESCONOCIDO"  # fallback para tipos no reconocidos
```

> 🔧 **Configurable**: Este enum debe poder extenderse fácilmente agregando nuevas entradas sin modificar el resto del pipeline. Para futuros años electorales, solo se actualiza este bloque.

#### 2.3 Validación cruzada con OCR

- Si el score de matching visual es < 0.6, usa el texto OCR como fallback: busca las keywords de cada tipo de boleta en el texto extraído.
- Lógica de decisión: `visual_match > 0.6` → usar resultado visual; de lo contrario, usar OCR.
- Si ninguno es concluyente → `TipoBoleta.DESCONOCIDO`.

**Output del módulo**: `TipoBoleta` (enum value) + `confidence_score` (float 0–1).

---

### MÓDULO 3 — EXTRACCIÓN DE CARACTERÍSTICAS (BACKBONES PREENTRENADOS)

**Objetivo**: Extraer representaciones vectoriales ricas de la imagen usando múltiples backbones de HuggingFace, luego fusionarlas.

#### 3.1 Backbones a usar (mínimo 3)

| # | Backbone | HuggingFace ID | Justificación |
|---|----------|---------------|---------------|
| 1 | **ResNet-50** | `microsoft/resnet-50` | Baseline sólido, robusto para documentos |
| 2 | **ViT-Base** | `google/vit-base-patch16-224` | Captura estructura global (layout) |
| 3 | **EfficientNet-B4** | `google/efficientnet-b4` | Eficiente, bueno para imágenes móviles |
| 4 *(opcional)* | **Swin Transformer** | `microsoft/swin-base-patch4-window7-224` | Jerarquía multiescala, ideal para documentos |

Todos deben cargarse con pesos preentrenados en ImageNet. **Congela todos los parámetros** (`requires_grad = False`). Solo se entrena la cabeza de clasificación.

#### 3.2 Fusión de características

```
ResNet-50   → vector 2048-d  ┐
ViT-Base    → vector 768-d   ├─► Concatenación → Proyección Linear → vector 512-d
EfficientNet → vector 1792-d ┘
```

- Concatena los tres vectores.
- Aplica una capa de proyección `Linear(total_dim, 512)` + `BatchNorm1d` + `ReLU`.
- Aplica `Dropout(0.3)` antes de pasar a la cabeza de clasificación.

**Output del módulo**: vector de características fusionado de 512 dimensiones.

---

### MÓDULO 4 — CLASIFICACIÓN MULTI-ETIQUETA

**Objetivo**: Clasificar el voto en partidos/coaliciones activos según el tipo de boleta, con soporte para activación simultánea (coaliciones).

#### 4.1 Etiquetas de clasificación

Define las siguientes clases como lista configurable:

```python
PARTIDOS = [
    "PAN",
    "PRI",
    "PRD",
    "PARTIDO_VERDE",
    "PARTIDO_DEL_TRABAJO",
    "MOVIMIENTO_CIUDADANO",
    "MORENA",
    "NULO"       # clase especial: voto nulo
]
# Total: 8 clases
```

#### 4.2 Coaliciones válidas (configurables por año)

```python
# Configuración electoral 2024 — modificar para otros años
COALICIONES_2024 = {
    "Fuerza y Corazón por México": ["PAN", "PRI", "PRD"],
    "Sigamos Haciendo Historia":   ["PARTIDO_VERDE", "PARTIDO_DEL_TRABAJO", "MORENA"],
}
```

> 🔧 **Configurable**: Para cambiar el año electoral, solo se actualiza el diccionario `COALICIONES_XXXX` y se pasa como parámetro al pipeline.

#### 4.3 Restricciones de activación simultánea

Implementa una capa de **post-procesamiento lógico** (no neuronal) que valide la salida:

- **Regla 1 — Coalición completa o nada**: Si se activa ≥2 partidos de una coalición, se activan **todos** los partidos de esa coalición.
- **Regla 2 — Exclusividad entre coaliciones**: No pueden activarse partidos de dos coaliciones distintas al mismo tiempo.
- **Regla 3 — NULO es exclusivo**: Si `NULO = 1`, todos los demás partidos se fuerzan a 0.
- **Regla 4 — Partidos independientes**: `MOVIMIENTO_CIUDADANO` puede activarse solo, sin pertenecer a ninguna coalición.

#### 4.4 Arquitectura de la cabeza de clasificación

```python
ClassificationHead(
    Linear(512, 256),
    ReLU(),
    Dropout(0.4),
    Linear(256, 8),
    Sigmoid()          # Multi-label: cada clase es independiente (0–1)
)
```

- **Función de pérdida**: `BCEWithLogitsLoss` con pesos por clase (para manejar desbalance).
- **Threshold de decisión**: 0.5 por defecto (configurable por clase).

#### 4.5 Condicionamiento por tipo de boleta

Según el `TipoBoleta` detectado en el Módulo 2, filtra las clases activas:

```python
CLASES_POR_BOLETA = {
    TipoBoleta.PRESIDENCIAL:        PARTIDOS,        # todas las clases activas
    TipoBoleta.DIPUTADOS_FEDERALES: PARTIDOS,        # todas las clases activas
    TipoBoleta.AYUNTAMIENTO:        PARTIDOS,        # todas las clases activas
    TipoBoleta.SINDICATURA:         PARTIDOS,        # todas las clases activas
    # Configurable: en algunos municipios ciertas boletas no incluyen todos los partidos
}
```

> 📌 Deja este mapeo **explícito y fácilmente editable** con un comentario que indique que debe actualizarse según la convocatoria electoral del año correspondiente.

**Output del módulo**: vector one-hot de 8 posiciones + nombre de coalición (si aplica) + `tipo_resultado` (`"VALIDO"` / `"NULO"`).

---

### MÓDULO 5 — GENERACIÓN DE DATOS SINTÉTICOS Y AUGMENTATION

**Objetivo**: A partir de las imágenes de ejemplo proporcionadas, generar un dataset suficiente para entrenar la cabeza de clasificación.

#### 5.1 Generación de imágenes sintéticas con PIL/OpenCV

A partir de cada imagen de ejemplo:

1. **Augmentation de imagen** (pre-backbone):
   - Rotaciones aleatorias ±15°
   - Variaciones de brillo y contraste (simula diferentes condiciones de luz)
   - Blur gaussiano leve (simula desenfoque de celular)
   - Ruido sal-y-pimienta
   - Recortes aleatorios (random crop 80–100%)
   - Volteo horizontal

2. Pasa cada imagen augmentada por los **3 backbones congelados** → obtén el vector fusionado de 512-d.

#### 5.2 Augmentation del vector de características (post-backbone)

Para multiplicar aún más los datos a nivel de embedding:

```python
def augment_feature_vector(vector: torch.Tensor, n_augmented: int = 20) -> list:
    augmented = []
    for _ in range(n_augmented):
        noise = torch.randn_like(vector) * 0.02          # ruido gaussiano leve
        scale = torch.FloatTensor(1).uniform_(0.95, 1.05) # variación de escala
        dropout_mask = torch.bernoulli(torch.ones_like(vector) * 0.95)  # dropout 5%
        aug = vector * scale * dropout_mask + noise
        augmented.append(aug)
    return augmented
```

#### 5.3 Etiquetado sintético

Genera etiquetas one-hot válidas para cada vector, respetando las reglas de coalición:

```python
# Distribución sugerida de etiquetas para el dataset sintético
DISTRIBUCION_ETIQUETAS = {
    "Fuerza y Corazón por México":  0.15,  # coalición PAN+PRI+PRD
    "Sigamos Haciendo Historia":    0.35,  # coalición VERDE+PT+MORENA
    "MOVIMIENTO_CIUDADANO":         0.20,  # partido independiente
    "PAN":                          0.05,  # partido solo
    "PRI":                          0.05,  # partido solo
    "PRD":                          0.03,
    "PARTIDO_VERDE":                0.03,
    "PARTIDO_DEL_TRABAJO":          0.03,
    "MORENA":                       0.03,
    "NULO":                         0.08,
}
```

**Output del módulo**: `Dataset` de PyTorch con pares `(vector_512d, label_8d_onehot)`.

---

### MÓDULO 6 — ENTRENAMIENTO

- **Optimizador**: `AdamW(lr=1e-3, weight_decay=1e-4)`
- **Scheduler**: `CosineAnnealingLR`
- **Epochs**: 50 (configurable)
- **Batch size**: 32
- **Split**: 70% train / 15% val / 15% test
- **Early stopping**: paciencia de 10 epochs sobre `val_hamming_loss`

---

### MÓDULO 7 — MÉTRICAS DE EVALUACIÓN MULTI-ETIQUETA

Implementa y reporta las siguientes métricas usando `sklearn.metrics`:

| Métrica | Función sklearn | Descripción |
|---------|----------------|-------------|
| **Hamming Loss** | `hamming_loss` | Fracción de etiquetas incorrectas (↓ mejor) |
| **LRAP** | `label_ranking_average_precision_score` | Precisión promedio de ranking de etiquetas (↑ mejor) |
| **Subset Accuracy** | `accuracy_score` | % de muestras con vector one-hot exactamente correcto |
| **Coverage Error** | `coverage_error` | Cuántas etiquetas hay que considerar para cubrir todas las verdaderas (↓ mejor) |

Genera un **reporte de evaluación** al final del entrenamiento con estas 4 métricas sobre el set de test.

---

### MÓDULO 8 — SALIDA JSON FINAL

**Objetivo**: Generar un JSON estructurado listo para ingestarse en un dashboard.

#### Schema del JSON de salida

```json
{
  "timestamp": "2024-06-02T14:35:22.456Z",
  "tipo_resultado": "VALIDO",
  "tipo_boleta": "PRESIDENCIAL",
  "resultado": {
    "coalicion": "Sigamos Haciendo Historia",
    "partidos": ["PARTIDO_VERDE", "PARTIDO_DEL_TRABAJO", "MORENA"],
    "one_hot": {
      "PAN": 0,
      "PRI": 0,
      "PRD": 0,
      "PARTIDO_VERDE": 1,
      "PARTIDO_DEL_TRABAJO": 1,
      "MOVIMIENTO_CIUDADANO": 0,
      "MORENA": 1,
      "NULO": 0
    }
  },
  "metadata": {
    "confidence_tipo_boleta": 0.94,
    "backbone_used": ["resnet50", "vit-base-patch16-224", "efficientnet-b4"],
    "pipeline_version": "1.0.0"
  }
}
```

**Reglas de construcción del JSON**:

- `timestamp`: momento exacto en que se llama al pipeline (no de entrenamiento).
- `tipo_resultado`: `"NULO"` si la clase NULO está activa; `"VALIDO"` en caso contrario.
- `coalicion`: nombre oficial de la coalición del diccionario `COALICIONES_2024`; `null` si el partido ganó solo.
- `partidos`: lista de partidos activos en el vector one-hot.
- `one_hot`: siempre presente con los 8 campos.
- `confidence_tipo_boleta`: score del Módulo 2.

---

## INSTRUCCIONES DE ENTREGA

Entrega lo siguiente en este orden:

### PARTE 1 — GUÍA TÉCNICA

Explica cada módulo con:
- Justificación de las decisiones de diseño.
- Diagrama del flujo completo del pipeline (en texto/ASCII o Mermaid).
- Consideraciones de escalabilidad para pasar de prototipo a producción.
- Riesgos técnicos identificados y cómo mitigarlos.

### PARTE 2 — CÓDIGO COMPLETO EJECUTABLE

Un único archivo Python `pipeline_boletas.py` (o notebook `pipeline_boletas.ipynb`) con:

```
pipeline_boletas.py
├── imports y configuración global
├── CONFIG dict  ← todos los parámetros configurables aquí
├── Módulo 1: PreprocessingModule
├── Módulo 2: HeaderDetectionModule
├── Módulo 3: FeatureExtractionModule
├── Módulo 4: MultiLabelClassifier
├── Módulo 5: SyntheticDataGenerator
├── Módulo 6: TrainingLoop
├── Módulo 7: EvaluationModule
├── Módulo 8: JSONOutputBuilder
└── main() ← orquesta todo el pipeline end-to-end
```

**Requisitos del código**:
- Ejecutable con `python pipeline_boletas.py --imagen ruta/a/imagen.jpg`
- `CONFIG` dict al inicio con **todos** los hiperparámetros y configuraciones editables.
- Docstrings en cada clase y método.
- Manejo de excepciones en cada módulo.
- Logging con `logging` estándar de Python.
- Compatible con CPU (no requiere GPU para el prototipo).

---

## RESTRICCIONES Y NOTAS FINALES

- ❌ No uses datos electorales reales ni imágenes con información personal identificable.
- ✅ Todas las imágenes de ejemplo proporcionadas se tratan como datos de referencia para generación sintética únicamente.
- ✅ El código debe ser reproducible: fija todas las semillas aleatorias (`torch.manual_seed`, `numpy.random.seed`).
- ✅ El archivo `requirements.txt` debe incluirse al final con todas las dependencias y versiones pinneadas.
- ✅ Incluye un bloque `if __name__ == "__main__"` que ejecute un demo end-to-end con datos sintéticos si no se pasa una imagen real.
