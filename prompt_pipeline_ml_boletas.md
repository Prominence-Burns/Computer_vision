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
3. **Extracción OCR**: Usa `pytesseract` con configuración `--psm 11` (sparse text, sin orden) sobre la `imagen_alta_res`. **No usar `--psm 6`** (bloque uniforme): las boletas son formularios con texto disperso, no bloques. Combinar con `image_to_data` para obtener bounding boxes y poder anclar texto↔posición (útil para R4 en Módulo 5.4 y para reconocer texto dentro de un recuadro específico). Para regiones específicas con texto denso conocido (encabezado, espacio CNR), invocar Tesseract sobre el crop con `--psm 6` (bloque) o `--psm 7` (línea única).
4. **Normalización de salida**: Texto limpio en mayúsculas, sin caracteres especiales, listo para el análisis de encabezado.

**Output del módulo** (dos artefactos, no uno):

| Artefacto | Resolución | Uso |
|---|---|---|
| `imagen_alta_res` | ≥1600 px en el lado largo (manteniendo aspect ratio), uint8 RGB | OCR (Tesseract), localización de recuadros (Módulo 2), crops por recuadro |
| `imagen_backbone` | 224×224×3 float32 normalizado (medias/desv. ImageNet) | Entrada a los backbones del Módulo 4 |
| `texto_ocr` | string | Texto extraído por Tesseract sobre `imagen_alta_res` |
| `regiones_ocr` | lista de bounding boxes + texto (de `image_to_data`) | Anclaje texto↔posición para reglas R4 |

> ⚠️ **No reducir el OCR a 224×224**: Tesseract necesita ~300 DPI sobre el texto para precisión razonable. Una boleta A4 a 224×224 garantiza fallo de OCR. Mantener dos pipelines separados es obligatorio.

---

### MÓDULO 2 — DETECCIÓN DE MARCAS

**Objetivo**: Identificar y caracterizar las marcas realizadas por el elector en cada recuadro de la boleta. Este módulo alimenta las reglas R1–R4 del AECC en el Módulo 5.

#### 2.1 Localización de recuadros

- Usando el resultado de la corrección geométrica del Módulo 1, detecta la cuadrícula de recuadros de la boleta mediante `cv2.findContours` + análisis de aspectRatio.
- Mapea cada recuadro detectado al ID del catálogo AECC:

```python
CATALOGO_RECUADROS = ["PAN", "PRI", "PRD", "PVEM", "PT", "MC", "MORENA", "CI", "SHH", "FCM", "CNR"]
```

#### 2.2 Análisis de marca por recuadro

Para cada recuadro localizado:

1. **Tipo de marca**: Clasifica el contenido usando descriptores de contorno y análisis de píxeles:
   - `cruz`: dos líneas que se cruzan en ángulo (~90°)
   - `raya`: línea o trazo simple
   - `palomita`: trazo en forma de ✓
   - `circulo`: trayectoria cerrada alrededor del recuadro
   - `texto`: presencia de caracteres detectados por OCR
   - `mancha`: área irregular sin patrón geométrico claro
   - `reflejo`: región brillante uniforme (artefacto de cámara)
   - `otro`: cualquier otro patrón

2. **Intensidad**: Basado en el valor medio de los píxeles en la región de la marca:
   - `clara`: contraste alto respecto al fondo
   - `tenue`: contraste bajo pero visible
   - `borrosa`: contraste muy bajo, límites difusos

3. **Posición**:
   - `dentro_recuadro`: `True` si el centroide de la marca está dentro del bounding box del recuadro.
   - `proporcion_en_recuadro`: fracción del área de la marca que se solapa con el recuadro (0.0–1.0).

#### 2.3 Estructura de salida

```python
@dataclass
class MarcaDetectada:
    recuadro:              str    # ID del catálogo AECC
    tipo_marca:            str    # cruz | raya | palomita | circulo | texto | mancha | reflejo | otro
    intensidad:            str    # clara | tenue | borrosa
    dentro_recuadro:       bool
    proporcion_en_recuadro: float  # 0.0 – 1.0
```

**Output del módulo**: lista de `MarcaDetectada` (un objeto por recuadro con marca detectada; recuadros sin marca no se incluyen).

---

### MÓDULO 3 — DETECCIÓN Y CLASIFICACIÓN DE ENCABEZADO

**Objetivo**: Identificar el **tipo de boleta** usando detección de características visuales (SIFT/ORB) y validación por OCR.

#### 3.0 Origen de los templates de referencia

El matching SIFT/ORB requiere un template limpio por cada `TipoBoleta`. **Dado que no contamos con dataset etiquetado real** (ver nota línea 20), los templates se construyen así, en este orden de preferencia:

1. **Render sintético del encabezado** a partir de la especificación INE: tipografía oficial + logo institucional + leyenda "BOLETA — ELECCIÓN DE …" + año. Se renderiza una imagen por `TipoBoleta` usando PIL + fuente conocida. Es la opción más limpia y reproducible.
2. **Crop manual de las imágenes de ejemplo**: si para algún tipo de boleta hay una imagen de referencia, recortar la franja superior limpia y guardarla como `templates/{tipo}.png`. Documentar como "template seed".
3. **Fallback OCR** cuando ningún template alcanza score ≥ 0.6 (ver 3.3).

> ⚠️ Con N=1 template por clase, el matching SIFT/ORB es brittle frente a iluminación, sombras y rotación residual. Por eso 3.3 es obligatorio, no opcional.

#### 3.1 Detección por características visuales

- Extrae la región superior de la imagen (primeros 10% de altura).
- Aplica **SIFT** (`cv2.SIFT_create()`) como detector primario.
- Aplica **ORB** (`cv2.ORB_create()`) como detector de respaldo (más rápido, sin patente).
- Usa matching por **FLANN** (para SIFT) o **BFMatcher** (para ORB) contra templates de referencia de cada tipo de boleta.

#### 3.2 Tipos de boleta válidos

Define los siguientes tipos como un `Enum` configurable:

```python
class TipoBoleta(Enum):
    PRESIDENCIA   = "presidencia"
    DIPUTACION_MR = "diputacion_mr"   # Mayoría Relativa
    DIPUTACION_RP = "diputacion_rp"   # Representación Proporcional
    SENADO_MR     = "senado_mr"
    SENADO_RP     = "senado_rp"
    MUNICIPAL     = "municipal"
    DESCONOCIDO   = "DESCONOCIDO"     # fallback para tipos no reconocidos
```

> 🔧 **Configurable**: Valores alineados con el campo `tipo_eleccion` del schema AECC (`urn:ine:aecc:casilla:v1`). Para futuros años electorales, solo se actualiza este bloque.

#### 3.3 Validación cruzada con OCR

- Si el score de matching visual es < 0.6, usa el texto OCR como fallback: busca las keywords de cada tipo de boleta en el texto extraído.
- Lógica de decisión: `visual_match > 0.6` → usar resultado visual; de lo contrario, usar OCR.
- Si ninguno es concluyente → `TipoBoleta.DESCONOCIDO`.

**Output del módulo**: `TipoBoleta` (enum value) + `confidence_score` (float 0–1).

---

### MÓDULO 4 — EXTRACCIÓN DE CARACTERÍSTICAS (BACKBONES PREENTRENADOS)

**Objetivo**: Extraer representaciones vectoriales ricas de la imagen usando múltiples backbones de HuggingFace, luego fusionarlas.

> 🧭 **Nota arquitectónica — papel de este módulo frente al Módulo 2**
>
> El campo `destinatario` del JSON AECC se determina **principalmente** por las marcas detectadas en cada recuadro (Módulo 2) + las reglas R1–R5 (Módulo 5.4). Los Módulos 4 y 5.5 actúan como **clasificador complementario** para casos ambiguos (marcas tenues, sin patrón claro) y como capa de validación cruzada. Si Módulo 2 produce un resultado de alta confianza, el clasificador neuronal aporta marginalmente; si Módulo 2 falla (recuadros no detectados, fotos muy degradadas), el clasificador neuronal es el fallback. Documentar este orden de precedencia en el código.
>
> 🟢 **Alternativa simplificada recomendada para prototipo (Opción A)**
>
> En lugar de usar 3 backbones globales sobre la imagen completa, considerar un **clasificador binario "marcado / no-marcado" por recuadro recortado**: un único backbone ligero (p. ej. `google/efficientnet-b0`, o incluso una CNN de 4 capas entrenada desde cero) recibe el crop de cada recuadro (~96×96 px tras corrección geométrica) y produce un score binario. La unión de los 11 scores binarios alimenta R1–R5 directamente, sin pasar por una cabeza multi-etiqueta de 9 clases. Esta ruta:
> - Resuelve el problema H3 (resolución para OCR ≠ resolución para backbones, ya separadas).
> - Reduce 3× el costo de inferencia.
> - Permite fine-tuning real del backbone sobre crops (mucho más cercanos al dominio).
> - Mantiene la coherencia con el output AECC.
>
> 🟡 **Alternativa ambiciosa (Opción B)**: un backbone único orientado a documentos estructurados (LayoutLMv3, Donut) o detección de objetos (YOLOv8/DETR) con clases = tipos de marca. Mantener Módulo 2 como decodificador clásico de respaldo.
>
> La especificación a continuación describe la **ruta original con 3 backbones globales** (3 ImageNet → fusión → 512-d → cabeza de 9 clases). Es válida pero subóptima para esta tarea; conservar sólo si hay restricción explícita de seguir esa arquitectura. En caso contrario, sustituir por Opción A.

#### 4.1 Backbones a usar (mínimo 3)

| # | Backbone | HuggingFace ID | Dim output | Resolución nativa | Justificación |
|---|----------|---------------|------------|-------------------|---------------|
| 1 | **ResNet-50** | `microsoft/resnet-50` | 2048 | 224×224 | Baseline sólido, robusto para documentos |
| 2 | **ViT-Base** | `google/vit-base-patch16-224` | 768 | 224×224 | Captura estructura global (layout) |
| 3 | **EfficientNet-B0** | `google/efficientnet-b0` | 1280 | 224×224 | Eficiente, **resolución nativa 224×224**. Sustituye al B4 original: B4 está diseñado para 380×380 y a 224 desperdicia capacidad. |
| 4 *(opcional)* | **Swin Transformer** | `microsoft/swin-base-patch4-window7-224` | 1024 | 224×224 | Jerarquía multiescala, ideal para documentos |

Todos deben cargarse con pesos preentrenados en ImageNet. **Congela todos los parámetros** (`requires_grad = False`). Solo se entrena la cabeza de clasificación.

> **Preprocesamiento per-backbone**: cada backbone tiene su `image_processor` propio en HuggingFace (medias/desviaciones, modo de resize). Usar `AutoImageProcessor.from_pretrained(model_id)` para cada uno; **no aplicar normalización global** sobre `imagen_backbone` antes de pasar al backbone — dejar que cada processor lo haga.

#### 4.2 Fusión de características

```
ResNet-50      → vector 2048-d ┐
ViT-Base       → vector  768-d ├─► Concatenación (4096-d)
EfficientNet-B0 → vector 1280-d ┘
                                 │
                                 ▼
                LayerNorm(4096) → Linear(4096, 512) → ReLU → Dropout(0.3)
                                 │
                                 ▼
                          vector 512-d
```

- Concatena los tres vectores: `total_dim = 2048 + 768 + 1280 = 4096`. **Si se cambian los backbones, actualizar `total_dim` en CONFIG**.
- Capa de proyección: `LayerNorm(total_dim) → Linear(total_dim, 512) → ReLU → Dropout(0.3)`.
- **Por qué `LayerNorm` y no `BatchNorm1d`**: BN estima media/varianza sobre el batch; con backbones congelados entrenados sobre dataset sintético, las estadísticas reflejarán artefactos sintéticos que no generalizan a fotos reales en inferencia. LayerNorm normaliza por muestra y es estable independientemente del tamaño/composición del batch.

**Output del módulo**: vector de características fusionado de 512 dimensiones.

---

### MÓDULO 5 — CLASIFICACIÓN MULTI-ETIQUETA

**Objetivo**: Clasificar el voto en partidos/coaliciones activos según el tipo de boleta, con soporte para activación simultánea (coaliciones).

#### 5.1 Etiquetas de clasificación

Define las siguientes clases como lista configurable:

```python
# IDs alineados con el catálogo de recuadros del AECC/INE 2024
PARTIDOS = [
    "PAN",
    "PRI",
    "PRD",
    "PVEM",    # Partido Verde Ecologista de México
    "PT",      # Partido del Trabajo
    "MC",      # Movimiento Ciudadano
    "MORENA",
    "CI",      # Candidatura Independiente
    "NULO",    # clase especial: voto nulo
]
# Total: 9 clases
```

#### 5.2 Coaliciones válidas (configurables por año)

```python
# Configuración electoral 2024 — modificar para otros años
# Clave: ID corto de coalición (usado como 'destinatario' en el AECC)
COALICIONES_2024 = {
    "FCM": {
        "nombre":   "Fuerza y Corazón por México",
        "partidos": ["PAN", "PRI", "PRD"],
    },
    "SHH": {
        "nombre":   "Sigamos Haciendo Historia",
        "partidos": ["PVEM", "PT", "MORENA"],
    },
}
```

> 🔧 **Configurable**: Para cambiar el año electoral, solo se actualiza el diccionario `COALICIONES_XXXX` y se pasa como parámetro al pipeline. El ID corto (`FCM`, `SHH`) es el valor que aparece en el campo `destinatario` del JSON AECC.

#### 5.3 Tipos y subtipos de clasificación (AECC)

Define los siguientes enums para que la salida sea conforme al schema `urn:ine:aecc:boleta:v1`:

```python
class TipoClasificacion(Enum):
    VALIDO        = "valido"
    NULO          = "nulo"
    NO_REGISTRADO = "no_registrado"

class SubtipoClasificacion(Enum):
    # Válidos (10)
    MARCA_ESTANDAR           = "marca_estandar"           # Art. 288 LGIPE
    MARCA_ATIPICA            = "marca_atipica"            # SUP-JIN-081/2006
    MARCA_FUERA_RECUADRO     = "marca_fuera_recuadro"     # SUP-JIN-021/2006
    RECUADRO_ENCERRADO       = "recuadro_encerrado"       # SUP-JIN-005/2006
    TEXTO_NO_OFENSIVO        = "texto_no_ofensivo"        # SUP-JIN-051/2012
    MULTIMARCA_POSITIVA      = "multimarca_positiva"      # SUP-JIN-011/2012
    NOMINATIVO_NOMBRE        = "nominativo_nombre"        # SUP-JIN-246/2006
    NOMINATIVO_APODO         = "nominativo_apodo"         # INE/CG517/2018
    COALICION_MULTIMARCA     = "coalicion_multimarca"     # Art. 288 pár.3 LGIPE
    MARCA_TENUE_PATRON       = "marca_tenue_patron"       # SUP-JIN-014/2012
    # Nulos (6)
    MARCA_TOTAL              = "marca_total"              # SM-JIN-046/2015
    INSULTO                  = "insulto"                  # SUP-JIN-069/2006
    MULTIMARCA_NO_COALIGADOS = "multimarca_no_coaligados" # SUP-JIN-028/2012
    BLANCO                   = "blanco"                   # Art. 291 pár.1 LGIPE
    ROTURA_GRAVE             = "rotura_grave"             # SUP-JIN-085/2006
    NOMINATIVO_CONTRADICTORIO= "nominativo_contradictorio"# INE/CG517/2018
    # No registrado (2)
    NOMBRE_CANDIDATO_NR      = "nombre_candidato_nr"      # SUP-JIN-246/2006
    SIGLAS_NR                = "siglas_nr"                # SM-JIN-046/2015
```

#### 5.4 Reglas de post-procesamiento lógico (AECC R1–R5)

Implementa una capa de **post-procesamiento lógico** (no neuronal) que aplique las reglas del AECC para determinar `destinatario`, `tipo`, `subtipo` y `requiere_revision`:

- **R1 — Coalición**: Si los recuadros marcados ⊆ partidos de una coalición → `destinatario = ID_coalición` (p. ej. `"SHH"`), `subtipo = COALICION_MULTIMARCA`.
- **R2 — Marca fuera de recuadro**: Si `proporcion_en_recuadro >= 0.5` en un solo recuadro pero `dentro_recuadro = False` → `valido / MARCA_FUERA_RECUADRO`. Si la proporción máxima < 0.5 → `requiere_revision = True`.
- **R3 — Marcas de rechazo**: Si exactamente un recuadro tiene marca positiva y los demás tienen marcas interpretables como rechazo → `valido / MULTIMARCA_POSITIVA`.
- **R4 — Texto**: Analiza `texto_detectado` del Módulo OCR: texto ofensivo → `nulo / INSULTO`; nombre de candidato registrado sin marca gráfica → `valido / NOMINATIVO_NOMBRE`; nombre + partido que no lo postuló → `nulo / NOMINATIVO_CONTRADICTORIO`; siglas/nombre desconocido → `no_registrado / SIGLAS_NR`.
- **R5 — Umbral de confianza para revisión humana**:
  - `confianza >= 0.85` → clasificación automática, `requiere_revision = False`
  - `0.60 ≤ confianza < 0.85` → clasificación tentativa, `requiere_revision = True`
  - `confianza < 0.60` → no clasificar, `requiere_revision = True`

```python
# Umbrales AECC R5 — incluir en CONFIG
CONFIANZA_AUTO     = 0.85
CONFIANZA_REVISION = 0.60
```

**Capa de mapping multi-etiqueta → `destinatario` único**:

El vector one-hot de 9 clases se convierte en un solo `destinatario` conforme al AECC:

```
one_hot (9 clases) + marcas_detectadas
    → aplicar R1–R5
    → destinatario: str | null   (ID partido, ID coalición, "CNR" o null)
       tipo:        TipoClasificacion
       subtipo:     SubtipoClasificacion
       confianza:   float
       requiere_revision: bool
```

El `one_hot` se conserva en el output como campo interno (`metadata.one_hot`) para auditoría, pero **no** es el campo principal de resultado.

**Tabla de fundamento legal** (incluir como constante en CONFIG):

```python
FUNDAMENTO_LEGAL = {
    "marca_estandar":            "Art. 288 LGIPE",
    "marca_atipica":             "SUP-JIN-081/2006",
    "marca_fuera_recuadro":      "SUP-JIN-021/2006",
    "recuadro_encerrado":        "SUP-JIN-005/2006",
    "texto_no_ofensivo":         "SUP-JIN-051/2012",
    "multimarca_positiva":       "SUP-JIN-011/2012",
    "nominativo_nombre":         "SUP-JIN-246/2006",
    "nominativo_apodo":          "INE/CG517/2018",
    "coalicion_multimarca":      "Art. 288 párrafo 3 LGIPE",
    "marca_tenue_patron":        "SUP-JIN-014/2012",
    "marca_total":               "SM-JIN-046/2015",
    "insulto":                   "SUP-JIN-069/2006",
    "multimarca_no_coaligados":  "SUP-JIN-028/2012",
    "blanco":                    "Art. 291 pár.1 inc.b) LGIPE – SUP-JIN-081/2006",
    "rotura_grave":              "SUP-JIN-085/2006",
    "nominativo_contradictorio": "INE/CG517/2018",
    "nombre_candidato_nr":       "SUP-JIN-246/2006",
    "siglas_nr":                 "SM-JIN-046/2015",
}
```

#### 5.5 Arquitectura de la cabeza de clasificación

```python
ClassificationHead(
    Linear(512, 256),
    ReLU(),
    Dropout(0.4),
    Linear(256, 9),    # 9 clases: PAN, PRI, PRD, PVEM, PT, MC, MORENA, CI, NULO
)
# Nota: NO aplicar Sigmoid en la cabeza. BCEWithLogitsLoss espera logits crudos
# (aplica sigmoide internamente con estabilidad numérica vía log-sum-exp trick).
# En inferencia: aplicar torch.sigmoid(logits) antes de comparar contra el threshold.
```

- **Función de pérdida**: `BCEWithLogitsLoss` con `pos_weight` por clase (ver más abajo) para manejar el desbalance del dataset.
- **`pos_weight` recomendado**: vector de 9 posiciones calculado como `(N_neg / N_pos)` por clase sobre el split de entrenamiento. Ejemplo con la distribución sugerida del Módulo 6.3, las clases minoritarias (`CI`, `PRD`, `PVEM`, `PT`, `MORENA` solas) reciben peso ≥10× para evitar colapso a "todo cero".

```python
# Cálculo en CONFIG, una vez fijado el dataset:
# pos_weight[i] = (n_muestras - n_positivas[i]) / max(n_positivas[i], 1)
pos_weight = torch.tensor([w_PAN, w_PRI, w_PRD, w_PVEM, w_PT, w_MC, w_MORENA, w_CI, w_NULO])
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

- **Alternativa válida si el desbalance persiste**: focal loss multi-etiqueta (γ=2) o `WeightedRandomSampler` en el DataLoader (oversampling estratificado).
- **Threshold de decisión**: 0.5 por defecto (configurable por clase en CONFIG).

#### 5.6 Condicionamiento por tipo de boleta

Según el `TipoBoleta` detectado en el Módulo 3 (HeaderDetection), filtra las clases activas:

```python
# Actualizar según convocatoria electoral del año: ciertos municipios/distritos pueden excluir
# partidos o coaliciones que no participan en esa elección específica.
# Ejemplo ilustrativo: un municipio donde no se registraron candidaturas independientes
# debería desactivar la clase "CI" para boletas municipales de ese municipio.
CLASES_POR_BOLETA = {
    TipoBoleta.PRESIDENCIA:   PARTIDOS,                          # todas activas
    TipoBoleta.DIPUTACION_MR: PARTIDOS,                          # todas activas
    TipoBoleta.DIPUTACION_RP: [p for p in PARTIDOS if p != "CI"],# RP no acepta independientes
    TipoBoleta.SENADO_MR:     PARTIDOS,
    TipoBoleta.SENADO_RP:     [p for p in PARTIDOS if p != "CI"],
    TipoBoleta.MUNICIPAL:     PARTIDOS,                          # ajustar por municipio en CONFIG
}
```

**Aplicación del filtro en inferencia (masking de logits)**:

```python
def aplicar_mascara_tipo_boleta(logits: torch.Tensor, tipo_boleta: TipoBoleta) -> torch.Tensor:
    """
    Pone -inf en las posiciones de clases inactivas para el tipo de boleta detectado.
    Tras sigmoid → probabilidad efectiva de 0 para esas clases.
    """
    clases_activas = set(CLASES_POR_BOLETA[tipo_boleta])
    mask = torch.tensor([1.0 if p in clases_activas else float("-inf") for p in PARTIDOS])
    return logits + mask
```

> 📌 Mantener este mapeo **explícito y fácilmente editable**. Debe actualizarse según la convocatoria electoral del año correspondiente. Si para algún año todos los tipos aceptan todas las clases, el masking es un no-op pero el código sigue siendo correcto.

**Output del módulo**: vector one-hot de 9 posiciones + `destinatario` (string único: ID de partido, `"FCM"`, `"SHH"`, `"CNR"` o `null`) + `TipoClasificacion` (`valido` / `nulo` / `no_registrado`) + `SubtipoClasificacion` + `confianza` (float) + `requiere_revision` (bool).

#### 5.7 Capa difusa opcional (FIS) — clasificador paralelo interpretable

**Objetivo**: Correr en paralelo al clasificador neuronal un **Sistema de Inferencia Difusa (FIS) Mamdani** que produzca, por cada recuadro, un grado de pertenencia continuo a la clase "voto válido", usando como entrada las features continuas del Módulo 2. Su propósito no es reemplazar la cabeza neuronal sino **dar trazabilidad jurídica** (cada regla difusa mapea explícitamente a un criterio del TEPJF) y servir de **segunda opinión** en casos de marca ambigua.

> 🟢 **Estatus**: opcional. Activable mediante `CONFIG["fis_enabled"] = True`. Si está desactivada, el pipeline funciona idéntico a 5.1–5.6. Si está activada, el FIS se ejecuta en paralelo y su salida se combina con la del clasificador neuronal según la política de fusión definida más abajo.

##### 5.7.1 Variables lingüísticas de entrada

Las features que produce el Módulo 2 (en su forma continua, no la etiqueta discretizada) se convierten en variables lingüísticas mediante funciones de pertenencia (membership functions, MFs) triangulares/trapezoidales:

| Variable de entrada | Origen (Módulo 2) | Términos lingüísticos | MFs sugeridas (sobre [0,1]) |
|---------------------|-------------------|-----------------------|------------------------------|
| `intensidad_norm` | Contraste medio normalizado de la marca respecto al fondo | `borrosa`, `tenue`, `clara` | trap(0, 0, 0.2, 0.4) / tri(0.3, 0.5, 0.7) / trap(0.6, 0.8, 1, 1) |
| `proporcion_en_recuadro` | Campo homónimo del `MarcaDetectada` | `baja`, `media`, `alta` | trap(0, 0, 0.2, 0.45) / tri(0.4, 0.6, 0.8) / trap(0.75, 0.9, 1, 1) |
| `geometricidad` | Score de ajuste del contorno al patrón geométrico esperado (cruz/palomita/raya) — calculado en 2.2 a partir de momentos de Hu | `irregular`, `parcial`, `nítida` | trap(0,0,0.25,0.5) / tri(0.4,0.6,0.8) / trap(0.7,0.9,1,1) |
| `centralidad` | 1 − distancia normalizada centroide-de-marca → centro-de-recuadro | `descentrada`, `media`, `centrada` | trap(0,0,0.3,0.5) / tri(0.4,0.6,0.8) / trap(0.7,0.9,1,1) |

> ⚠️ **Calibración de MFs**: los breakpoints anteriores son seed. Con dataset sintético deben re-ajustarse observando los histogramas de cada feature sobre el conjunto generado en Módulo 6. Fijar las MFs sobre artefactos del augmentation es el riesgo principal — documentar en el reporte de evaluación los percentiles reales que sustentan cada MF.

##### 5.7.2 Variable lingüística de salida

| Variable de salida | Términos | MFs sobre [0,1] |
|--------------------|----------|------------------|
| `voto_valido_grado` | `nulo`, `dudoso`, `tentativo`, `fuerte` | trap(0,0,0.1,0.3) / tri(0.2,0.4,0.6) / tri(0.5,0.7,0.85) / trap(0.8,0.9,1,1) |

Defuzzificación por **centroide** (`centroid` / centro de gravedad). El escalar resultante ∈ [0,1] es el `fis_score` por recuadro.

##### 5.7.3 Base de reglas (mapeo TEPJF → reglas difusas)

Cada regla referencia el criterio jurídico que la sustenta, para que el campo `fundamento_legal` del JSON AECC pueda construirse trazando qué reglas dispararon:

```text
R_fis_1  SI intensidad ES clara      Y proporcion ES alta    Y geometricidad ES nítida      ENTONCES voto ES fuerte       [Art. 288 LGIPE]
R_fis_2  SI intensidad ES clara      Y proporcion ES alta    Y geometricidad ES parcial     ENTONCES voto ES fuerte       [SUP-JIN-081/2006: marca atípica]
R_fis_3  SI intensidad ES tenue      Y proporcion ES alta    Y centralidad ES centrada      ENTONCES voto ES tentativo    [SUP-JIN-014/2012: marca tenue]
R_fis_4  SI proporcion ES media      Y centralidad ES media                                  ENTONCES voto ES tentativo    [SUP-JIN-021/2006: fuera de recuadro]
R_fis_5  SI proporcion ES baja                                                              ENTONCES voto ES nulo         [marca insuficiente]
R_fis_6  SI intensidad ES borrosa    Y geometricidad ES irregular                            ENTONCES voto ES dudoso       [requiere revisión humana]
R_fis_7  SI geometricidad ES nítida  Y centralidad ES descentrada Y proporcion ES alta      ENTONCES voto ES tentativo    [SUP-JIN-021/2006]
R_fis_8  SI intensidad ES clara      Y geometricidad ES nítida   Y proporcion ES media       ENTONCES voto ES tentativo    [SUP-JIN-005/2006: recuadro encerrado]
```

Operadores:
- T-norma (Y lógico): `min`
- S-norma (O lógico): `max`
- Implicación: `min` (Mamdani)
- Agregación: `max`
- Defuzzificación: `centroid`

Implementación sugerida con `scikit-fuzzy` (`skfuzzy.control`) — añadir a `requirements.txt`. Como alternativa pura-Python sin dependencias extra, una implementación manual de ~80 líneas es suficiente y permite fijar la versión exactamente.

##### 5.7.4 Política de fusión FIS ⊕ Neuronal

Por cada recuadro `r ∈ CATALOGO_RECUADROS` se obtienen dos scores en [0,1]:

- `neural_score[r]`: `torch.sigmoid(logits[idx(r)])` de la cabeza del Módulo 5.5 (sólo para los 9 índices de `PARTIDOS`; recuadros como `SHH`/`FCM`/`CNR` no tienen score neuronal directo).
- `fis_score[r]`: salida del FIS aplicado a las features del Módulo 2 para ese recuadro.

Fusión configurable en CONFIG:

```python
CONFIG["fis_fusion_mode"] = "weighted"   # uno de: "weighted" | "min" | "veto" | "report_only"
CONFIG["fis_weight"]      = 0.35         # peso del FIS si fusion_mode="weighted"; neural pesa (1 - fis_weight)
```

| Modo | Comportamiento | Cuándo usarlo |
|------|----------------|---------------|
| `report_only` | El FIS se calcula y se persiste en `metadata.fis`, pero **no** afecta la decisión. Modo por defecto para auditoría. | Demostrar viabilidad sin comprometer el clasificador principal. |
| `weighted` | `score_final[r] = (1 − w)·neural_score[r] + w·fis_score[r]` con `w = CONFIG["fis_weight"]`. El score fusionado reemplaza al neuronal antes de aplicar el threshold 0.5. | Cuando la calibración de MFs es confiable. |
| `min` | `score_final[r] = min(neural_score[r], fis_score[r])`. Conservador: requiere acuerdo de ambos clasificadores para marcar voto. | Maximizar precisión a costa de recall. Útil si el costo de un falso positivo es alto. |
| `veto` | El FIS sólo puede **rebajar** el resultado neuronal a `requiere_revision = True` cuando `fis_score[r] < CONFIG["fis_veto_threshold"]` (sugerido 0.35) y el neuronal lo marcó como voto. Nunca lo eleva. | Capa de seguridad jurídica: si el FIS no encuentra fundamento en TEPJF, la boleta va a revisión humana aunque la red esté segura. |

##### 5.7.5 Integración con R1–R5

El FIS **no reemplaza R1–R5**; se ejecuta antes y produce un `marcas_detectadas_enriquecido` donde cada `MarcaDetectada` lleva un campo adicional `fis_score: float`. Las reglas R1–R5 operan igual, pero:

- **R5 (umbral de confianza)** consume `score_final` (resultado de la fusión) en vez del score neuronal puro.
- **R2 (marca fuera de recuadro)** puede usar `fis_score` para distinguir entre marca-fuera-válida (`fis_score` alto sobre `centralidad descentrada + proporcion alta`) y mancha accidental (`fis_score` bajo).
- Las reglas que dispararon en el FIS (`R_fis_*`) se loguean para que `fundamento_legal` en el JSON AECC pueda citarlas explícitamente.

##### 5.7.6 Campos añadidos al output

Al `MarcaDetectada` (Módulo 2.3) se le agrega — sólo si `CONFIG["fis_enabled"] = True`:

```python
@dataclass
class MarcaDetectada:
    # ... campos previos ...
    fis_score:          float | None = None  # grado de pertenencia "voto válido" ∈ [0,1]
    fis_rules_fired:    list[str] | None = None  # IDs de reglas difusas que dispararon (p. ej. ["R_fis_2","R_fis_8"])
```

Al JSON AECC de boleta (Módulo 9) se le agrega bajo `metadata`:

```json
"metadata": {
  "fis": {
    "enabled":     true,
    "fusion_mode": "weighted",
    "fis_weight":  0.35,
    "per_recuadro": {
      "PVEM": { "fis_score": 0.87, "rules_fired": ["R_fis_1"] },
      "PT":   { "fis_score": 0.84, "rules_fired": ["R_fis_2"] }
    }
  }
}
```

##### 5.7.7 Riesgos específicos

- **Calibración sobre datos sintéticos**: las MFs se ajustan a los histogramas del augmentation, no a fotos reales → reportar en el evaluation report la divergencia entre el `fis_score` medio sobre sintético vs. sobre las imágenes de ejemplo originales.
- **Doble contabilización de incertidumbre**: si tanto el neuronal como el FIS están mal calibrados en la misma dirección, fusionarlos amplifica el error. Mitigación: arrancar con `fusion_mode = "report_only"` y promover a `weighted` sólo cuando el reporte de Módulo 8 muestre que el FIS aporta señal independiente (correlación con el neuronal < 0.85).
- **Costo de mantenimiento**: las reglas son código vivo que debe revisarse cada vez que cambia jurisprudencia del TEPJF. Documentar en CONFIG la versión del corpus jurisprudencial usado (`CONFIG["fis_jurisprudence_version"] = "2024-Q2"`).

---

### MÓDULO 6 — GENERACIÓN DE DATOS SINTÉTICOS Y AUGMENTATION

**Objetivo**: A partir de las imágenes de ejemplo proporcionadas, generar un dataset suficiente para entrenar la cabeza de clasificación.

#### 6.1 Generación de imágenes sintéticas con PIL/OpenCV

A partir de cada imagen de ejemplo:

1. **Augmentation de imagen** (pre-backbone):
   - Rotaciones aleatorias ±15°
   - Variaciones de brillo y contraste (simula diferentes condiciones de luz)
   - Blur gaussiano leve (simula desenfoque de celular)
   - Ruido sal-y-pimienta
   - Recortes aleatorios (random crop 80–100%)
   - Volteo horizontal

2. Pasa cada imagen augmentada por los **3 backbones congelados** → obtén el vector fusionado de 512-d.

#### 6.2 Augmentation del vector de características (post-backbone) — OPCIONAL Y RESTRINGIDO

> ⚠️ **Advertencia**: perturbar embeddings ya extraídos genera muestras fuera de la variedad natural de imágenes de boletas. La cabeza aprende a invariar a ruido sintético en el espacio latente, no a variabilidad real del dominio. **Si es posible, preferir augmentation en el espacio imagen (6.1) y omitir 6.2 completamente**.
>
> Si se conserva 6.2, aplicar **sólo a las muestras de train** (nunca val/test, para evitar leakage), y reducir el dropout-mask al 1% (al 5% original se mataban 25 de las 512 dimensiones, demasiado agresivo).

```python
def augment_feature_vector(vector: torch.Tensor, n_augmented: int = 5) -> list:
    """
    Solo se invoca sobre muestras del split de train. NUNCA sobre val/test.
    """
    augmented = []
    for _ in range(n_augmented):
        noise = torch.randn_like(vector) * 0.02            # ruido gaussiano leve
        scale = torch.FloatTensor(1).uniform_(0.97, 1.03)  # variación de escala
        dropout_mask = torch.bernoulli(torch.ones_like(vector) * 0.99)  # dropout 1%
        aug = vector * scale * dropout_mask + noise
        augmented.append(aug)
    return augmented
```

#### 6.3 Etiquetado sintético

Genera etiquetas one-hot válidas para cada vector, respetando las reglas de coalición:

```python
# Distribución sugerida de etiquetas para el dataset sintético
# Claves: ID de partido/coalición del catálogo AECC/INE 2024
DISTRIBUCION_ETIQUETAS = {
    "FCM":    0.15,  # coalición PAN+PRI+PRD (Fuerza y Corazón por México)
    "SHH":    0.35,  # coalición PVEM+PT+MORENA (Sigamos Haciendo Historia)
    "MC":     0.20,  # partido independiente (Movimiento Ciudadano)
    "PAN":    0.05,  # partido solo
    "PRI":    0.05,  # partido solo
    "PRD":    0.03,
    "PVEM":   0.03,
    "PT":     0.03,
    "MORENA": 0.03,
    "CI":     0.01,  # candidatura independiente
    "NULO":   0.07,
}
```

**Mapeo de IDs de coalición al one-hot de 9 clases**: las claves `"FCM"` y `"SHH"` en `DISTRIBUCION_ETIQUETAS` representan coaliciones, pero el clasificador del Módulo 5 emite sólo 9 clases (`PAN…NULO`) sin entradas para `FCM`/`SHH`. La conversión es:

```python
def coalition_to_onehot(destinatario_id: str) -> torch.Tensor:
    """
    Convierte un ID de destinatario (partido individual, coalición o NULO)
    al vector one-hot de 9 posiciones que consume el clasificador.
    Para coaliciones activa todos los partidos miembros.
    """
    onehot = torch.zeros(len(PARTIDOS))  # 9 ceros
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
```

Adicionalmente, genera `marcas_detectadas` sintéticas para cada muestra (tipo de marca, intensidad y posición por recuadro), respetando la distribución de etiquetas. Para una muestra con `destinatario="SHH"` deben generarse marcas en los recuadros `PVEM`, `PT` y `MORENA` (las tres simultáneamente) para que R1 (coalición) dispare correctamente en el Módulo 5.

**Output del módulo**: `Dataset` de PyTorch con tuplas `(vector_512d, label_9d_onehot, source_image_id, marcas_detectadas)`. El campo `source_image_id` (`str`, identificador de la imagen original de la cual se derivó la muestra augmentada) es **obligatorio** para que el split del Módulo 7 pueda agrupar correctamente con `GroupShuffleSplit` y evitar data leakage.

---

### MÓDULO 7 — ENTRENAMIENTO

- **Optimizador**: `AdamW(lr=1e-3, weight_decay=1e-4)`
- **Scheduler**: `CosineAnnealingLR(T_max=num_epochs, eta_min=1e-6)` — `T_max=50` con la configuración por defecto.
- **Epochs**: 50 (configurable)
- **Batch size**: 32
- **Split**: 70% train / 15% val / 15% test, **agrupado por `source_image_id`** mediante `sklearn.model_selection.GroupShuffleSplit`. Todas las augmentaciones derivadas de una misma imagen fuente deben caer en el mismo split; lo contrario produce data leakage e infla artificialmente las métricas.
- **Early stopping**: paciencia de 10 epochs sobre `val_hamming_loss`. **Nota**: con datasets sintéticos pequeños la convergencia suele ocurrir en <15 epochs; ajustar paciencia y `num_epochs` empíricamente.

---

### MÓDULO 8 — MÉTRICAS DE EVALUACIÓN

#### 8.1 Métricas primarias (lo que realmente importa para el AECC)

| Métrica | Cómo se calcula | Por qué es primaria |
|---------|-----------------|---------------------|
| **Exact destinatario accuracy** | `mean(destinatario_pred == destinatario_true)` tras aplicar R1–R5 sobre `marcas_detectadas` + cabeza neuronal | Es el campo que efectivamente se persiste en `ballots.detected_vote`. Si la cabeza acierta el one-hot pero R1–R5 colapsan mal a coalición, la métrica falla — exactamente como queremos. |
| **F1 macro (9 clases)** | `f1_score(y_true, y_pred, average="macro", zero_division=0)` | Multi-etiqueta con desbalance fuerte (35% SHH, 1% CI). Macro F1 penaliza colapso a clases mayoritarias. |
| **F1 micro (9 clases)** | `f1_score(y_true, y_pred, average="micro")` | Refleja desempeño agregado, complementa el macro. |
| **Confusion matrix por `subtipo` AECC** | Matriz 18×18 de `subtipo_pred` × `subtipo_true` | Detecta confusiones específicas (p. ej. `marca_estandar` ↔ `marca_fuera_recuadro`). Permite ver el desempeño por regla R1–R5. |
| **Tasa de `requiere_revision`** | `mean(requiere_revision_pred)` | Mide el costo operativo: cuántas boletas necesitarán revisión humana. Si excede ~10%, el sistema no escala. |

#### 8.2 Métricas secundarias (diagnóstico del clasificador neuronal)

| Métrica | Función sklearn | Descripción |
|---------|----------------|-------------|
| Hamming Loss | `hamming_loss` | Fracción de etiquetas incorrectas (↓ mejor) |
| LRAP | `label_ranking_average_precision_score` | Precisión promedio de ranking (↑ mejor) |
| Subset Accuracy | `accuracy_score` | % de muestras con one-hot exactamente correcto |
| Coverage Error | `coverage_error` | Cuántas etiquetas hay que considerar para cubrir todas las verdaderas (↓ mejor) |

> ⚠️ **No usar las secundarias como métricas primarias**. En multi-etiqueta de 9 clases con la mayoría de posiciones en 0, un clasificador trivial "todo cero" obtiene Hamming Loss baja y Subset Accuracy alta sobre votos blancos.

#### 8.3 Salud del dataset sintético

Métrica auxiliar para detectar sobreajuste a artefactos sintéticos:

```python
def health_check_sintetico(embeddings_sinteticos, embeddings_reales) -> float:
    """
    Distancia coseno media entre el centroide de embeddings sintéticos
    y los embeddings de las imágenes reales de ejemplo. Cuanto más alta,
    mayor el dominio gap → mayor riesgo de overfit a sintético.
    Reportar al final del entrenamiento.
    """
    centroide_sint = embeddings_sinteticos.mean(dim=0)
    return (1 - F.cosine_similarity(centroide_sint.unsqueeze(0), embeddings_reales)).mean().item()
```

#### 8.4 Reporte de evaluación

Genera un **reporte de evaluación** al final del entrenamiento con todas las métricas primarias + secundarias + health check sobre el set de test, en formato JSON serializable para auditoría.

---

### MÓDULO 9 — JSON DE BOLETA (AECC Parte 1)

**Objetivo**: Generar el JSON de boleta conforme al schema `urn:ine:aecc:boleta:v1`, listo para persistirse en la entidad `ballots` de la base de datos PREP e ingestarse en el dashboard.

#### Schema del JSON de boleta

```json
{
  "$schema": "urn:ine:aecc:boleta:v1",
  "boleta_id":  "B-00134",
  "casilla_id": "NL-01-865-B",
  "timestamp":  "2024-06-02T14:35:22.456Z",
  "image_url":  "ruta/a/imagen_original.jpg",

  "clasificacion": {
    "tipo":               "valido",
    "subtipo":            "coalicion_multimarca",
    "destinatario":       "SHH",
    "confianza":          0.91,
    "requiere_revision":  false
  },

  "marcas_detectadas": [
    {
      "recuadro":              "PVEM",
      "tipo_marca":            "cruz",
      "intensidad":            "clara",
      "dentro_recuadro":       true,
      "proporcion_en_recuadro": 0.88
    },
    {
      "recuadro":              "PT",
      "tipo_marca":            "cruz",
      "intensidad":            "clara",
      "dentro_recuadro":       true,
      "proporcion_en_recuadro": 0.85
    }
  ],

  "anomalias": [],

  "texto_detectado": null,

  "fundamento_legal": "Art. 288 párrafo 3 LGIPE — marcas en recuadros de partidos coaligados",

  "metadata": {
    "backbone_used":        ["resnet50", "vit-base-patch16-224", "efficientnet-b4"],
    "pipeline_version":     "1.0.0",
    "tipo_boleta":          "diputacion_mr",
    "one_hot_interno": {
      "PAN": 0, "PRI": 0, "PRD": 0, "PVEM": 1,
      "PT": 1, "MC": 0, "MORENA": 0, "CI": 0, "NULO": 0
    }
  }
}
```

**Reglas de construcción del JSON**:

- `boleta_id`: recibido como parámetro (o generado como UUID si no se pasa).
- `casilla_id`: recibido como parámetro.
- `timestamp`: momento exacto de la llamada al pipeline.
- `image_url`: ruta de la imagen procesada (parámetro `--imagen`).
- `clasificacion.tipo`: resultado de `TipoClasificacion` tras aplicar R1–R5.
- `clasificacion.subtipo`: resultado de `SubtipoClasificacion`.
- `clasificacion.destinatario`: ID único del partido/coalición ganador, `"CNR"` para candidato no registrado, o `null` si es nulo sin destinatario.
- `clasificacion.confianza`: score de confianza del clasificador.
- `clasificacion.requiere_revision`: determinado por R5 (umbral 0.85 / 0.60).
- `fundamento_legal`: extraído de la tabla `FUNDAMENTO_LEGAL` según el `subtipo`.
- `metadata.one_hot_interno`: vector one-hot de 9 posiciones (campo auxiliar para auditoría, no es el resultado oficial).

**Mapeo a entidad `ballots` (base de datos PREP)**:

| Campo JSON (`boleta`) | Campo DB (`ballots`) |
|-----------------------|----------------------|
| `boleta_id` | `id` |
| `casilla_id` | `polling_station_id` |
| `image_url` | `image_url` |
| `clasificacion.destinatario` | `detected_vote` |
| `clasificacion.confianza` | `confidence_score` |
| `clasificacion.requiere_revision` | `reviewed_by_human` |
| `clasificacion.tipo` + `subtipo` | `final_classification` |
| `timestamp` | `created_at` |

**Eventos que debe emitir este módulo** (para la entidad `events` del PREP):

```python
EVENTOS_PIPELINE = {
    "ballot_scanned":        "siempre al procesar una boleta",
    "vote_detected":         "si clasificacion.tipo != None",
    "inconsistency_detected":"si requiere_revision=True o anomalias no vacío",
}
```

---

### MÓDULO 10 — JSON DE CASILLA / AECC (Parte 2)

**Objetivo**: Acumular los JSONs de boletas individuales y generar el JSON de casilla conforme al schema `urn:ine:aecc:casilla:v1`. Este JSON representa el Acta de Escrutinio y Cómputo de Casilla (AECC).

**Input**: lista de JSONs de boletas + metadatos administrativos de la casilla (proporcionados por el funcionario de casilla; el clasificador no los genera):

```python
metadatos_casilla = {
    "casilla_id": "NL-01-865-B",
    "entidad_federativa": "Nuevo León",
    "municipio_o_delegacion": "Monterrey",
    "distrito": "01",
    "seccion": "865",
    "tipo_casilla": "basica",      # basica | contigua | extraordinaria | especial
    "tipo_eleccion": "diputacion_mr",
    "proceso_electoral": "2023-2024",
    # Datos capturados por el funcionario (no por el clasificador):
    "boletas_recibidas": 750,
    "BS": 166,    # boletas sobrantes
    "PV": 580,    # personas que votaron en lista nominal
    "RPPV": 4,    # representantes de partido votaron fuera de lista
}
```

**Lógica de agregación**:

1. Cuenta votos por `destinatario` → construye `bloque_2.resultados`.
2. Suma votos nulos (`tipo = "nulo"`) → `bloque_2.VN`.
3. Suma candidatos no registrados (`tipo = "no_registrado"`) → `bloque_2.CNR`.
4. Calcula `RV = sum(votos) + CNR + VN`.
5. Verifica los 4 criterios de consistencia del AECC:
   - `criterio_1`: `PV + RPPV == SV`
   - `criterio_2`: `SV == BSU` (BSU = total de boletas en urna = boletas clasificadas)
   - `criterio_3`: `BSU == RV`
   - `criterio_4`: `sum(resultados[].votos) + CNR + VN == RV`
6. Genera `hash_boletas`: SHA-256 sobre la **serialización canónica** del array de boletas. La serialización JSON estándar no es determinista (orden de claves, separadores, escapes Unicode) → hashes distintos para el mismo contenido. Usar exactamente:

```python
import json, hashlib
boletas_ordenadas = sorted(lista_boletas, key=lambda b: b["boleta_id"])
canon = json.dumps(
    boletas_ordenadas,
    sort_keys=True,
    separators=(",", ":"),   # sin espacios
    ensure_ascii=False,       # preserva UTF-8
).encode("utf-8")
hash_boletas = hashlib.sha256(canon).hexdigest()
```

  Cualquier desviación de estos parámetros invalida la verificación de integridad cross-corrida.

**Schema del JSON de casilla (AECC)**:

```json
{
  "$schema": "urn:ine:aecc:casilla:v1",
  "metadatos": { ... },
  "bloque_1": {
    "boletas_recibidas": 750,
    "BS": 166,
    "PV": 580,
    "RPPV": 4,
    "SV": 584,
    "BSU": 584
  },
  "bloque_2": {
    "resultados": [
      { "partido_o_coalicion": "Sigamos Haciendo Historia", "id": "SHH",
        "es_coalicion": true, "partidos_coalicion": ["PVEM","PT","MORENA"], "votos": 73 },
      { "partido_o_coalicion": "Movimiento Ciudadano", "id": "MC",
        "es_coalicion": false, "partidos_coalicion": null, "votos": 91 }
    ],
    "CNR": 3,
    "VN": 40,
    "RV": 584
  },
  "consistencia": {
    "criterio_1_pv_rppv_sv": true,
    "criterio_2_sv_bsu": true,
    "criterio_3_bsu_rv": true,
    "criterio_4_sum_vi_rv": true,
    "acta_consistente": true,
    "tipo_error": "ninguno"
  },
  "incidentes": { "se_presentaron": false, "descripcion": null, "hojas_de_incidentes": 0 },
  "boletas_procesadas": 584,
  "boletas_revision_humana": 7,
  "hash_boletas": "<SHA-256>"
}
```

**Mapeo a entidades de la base de datos PREP**:

| Campo JSON (casilla) | Entidad DB |
|----------------------|-----------|
| JSON de casilla completo | `tally_sheets` (`extracted_text`, `validation_status`, `total_votes`, `null_votes`) |
| `bloque_2.resultados[]` | `results` (un registro por partido/coalición) |
| Evento emitido: `results_submitted` | `events` |

> ⚠️ Si `consistencia.acta_consistente = false`, el sistema debe marcar el acta para revisión antes de enviarla al sistema de cómputo distrital.

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
│     (PARTIDOS, COALICIONES_XXXX, TipoBoleta, FUNDAMENTO_LEGAL,
│      CONFIANZA_AUTO, CONFIANZA_REVISION, EVENTOS_PIPELINE, ...)
├── Módulo 1:  PreprocessingModule       ← imagen → imagen preprocesada + OCR
├── Módulo 2:  MarkDetectionModule       ← imagen preprocesada → marcas_detectadas
├── Módulo 3:  HeaderDetectionModule     ← → TipoBoleta + confidence_score
├── Módulo 4:  FeatureExtractionModule   ← → vector 512-d
├── Módulo 5:  MultiLabelClassifier      ← → one_hot + destinatario + tipo + subtipo
├── Módulo 6:  SyntheticDataGenerator
├── Módulo 7:  TrainingLoop
├── Módulo 8:  EvaluationModule
├── Módulo 9:  JSONOutputBuilder         ← → JSON AECC de boleta (urn:ine:aecc:boleta:v1)
├── Módulo 10: CasillaAggregator         ← → JSON AECC de casilla (urn:ine:aecc:casilla:v1)
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
- ✅ El código debe ser reproducible. Incluir en CONFIG el helper:

```python
def set_global_seed(seed: int = 42) -> None:
    import os, random
    import numpy as np
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Activa modo determinístico estricto (requiere CUBLAS_WORKSPACE_CONFIG=:4096:8 en CUDA):
    torch.use_deterministic_algorithms(True, warn_only=True)
```

  Invocar `set_global_seed(CONFIG["seed"])` antes de instanciar modelo, optimizador o DataLoaders. Si se usa Albumentations o transforms estocásticos adicionales, fijar también su `seed` argument.
- ✅ El archivo `requirements.txt` debe incluirse al final con todas las dependencias y versiones pinneadas.
- ✅ Incluye un bloque `if __name__ == "__main__"` que ejecute un demo end-to-end con datos sintéticos si no se pasa una imagen real.
