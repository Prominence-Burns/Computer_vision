# Entidades principales de la base de datos del sistema PREP con IA

## 1. `ballots`

### ¿Qué es?
La entidad `ballots` representa cada boleta electoral procesada por el sistema de visión computacional.

Cada registro almacena la información relacionada con:
- la imagen de la boleta,
- la clasificación realizada por IA,
- la validación humana,
- y el resultado final asociado.

---

### ¿Para qué sirve?
Esta entidad permite:

- Registrar digitalmente cada boleta escaneada.
- Detectar votos válidos, nulos o inconsistentes.
- Conservar evidencia visual para auditorías.
- Medir la confianza del modelo de IA.
- Permitir revisiones manuales en casos ambiguos.

---

### Información importante que puede almacenar

| Campo | Descripción |
|---|---|
| `id` | Identificador único de la boleta |
| `polling_station_id` | Casilla donde fue procesada |
| `image_url` | Ruta o URL de la imagen |
| `detected_vote` | Clasificación realizada por IA |
| `confidence_score` | Nivel de confianza de la predicción |
| `reviewed_by_human` | Indica si fue validada manualmente |
| `final_classification` | Resultado definitivo |
| `created_at` | Timestamp de creación |

---

# 2. `results`

### ¿Qué es?
La entidad `results` almacena los resultados agregados del conteo electoral.

Representa el consolidado de votos por:
- partido,
- candidato,
- casilla,
- o distrito.

---

### ¿Para qué sirve?
Permite:

- Mostrar resultados en el dashboard.
- Generar estadísticas en tiempo real.
- Comparar resultados preliminares y oficiales.
- Visualizar avances del PREP.
- Realizar análisis posteriores.

---

### Información importante que puede almacenar

| Campo | Descripción |
|---|---|
| `id` | Identificador único |
| `polling_station_id` | Casilla relacionada |
| `party` | Partido político |
| `vote_count` | Cantidad de votos |
| `source` | Fuente del resultado |
| `created_at` | Timestamp del registro |

---

# 3. `tally_sheets`

### ¿Qué es?
La entidad `tally_sheets` representa las actas de escrutinio y cómputo utilizadas por el PREP.

Las actas son el documento oficial generado al finalizar el conteo en casilla.

---

### ¿Para qué sirve?
Permite:

- Digitalizar las actas físicas.
- Extraer texto mediante OCR.
- Validar resultados automáticamente.
- Relacionar boletas con resultados oficiales.
- Generar trazabilidad documental.

---

### Información importante que puede almacenar

| Campo | Descripción |
|---|---|
| `id` | Identificador del acta |
| `polling_station_id` | Casilla asociada |
| `image_url` | Imagen del acta |
| `extracted_text` | Texto extraído mediante OCR |
| `validation_status` | Estado de validación |
| `total_votes` | Total de votos registrados |
| `null_votes` | Votos nulos registrados |
| `created_at` | Timestamp del registro |

---

# 4. `users`

### ¿Qué es?
La entidad `users` representa a las personas que interactúan con el sistema.

Incluye:
- funcionarios de casilla,
- operadores,
- auditores,
- administradores,
- y observadores.

---

### ¿Para qué sirve?
Permite:

- Controlar accesos al sistema.
- Identificar quién realizó cada acción.
- Mantener trazabilidad de operaciones.
- Registrar actividad de usuarios.
- Implementar autenticación y permisos.

---

### Información importante que puede almacenar

| Campo | Descripción |
|---|---|
| `id` | Identificador único del usuario |
| `name` | Nombre del usuario |
| `role` | Rol dentro del sistema |
| `polling_station_id` | Casilla asignada |
| `created_at` | Fecha de registro |

---

# 5. `events`

### ¿Qué es?
La entidad `events` es el núcleo del sistema de auditoría.

Registra cada acción realizada dentro del sistema como un evento estructurado.

Esta entidad implementa el concepto de:

```text
Event Sourcing
```

---

### ¿Para qué sirve?
Permite:

- Construir una bitácora auditable.
- Registrar trazabilidad completa.
- Reconstruir operaciones históricas.
- Identificar quién hizo qué y cuándo.
- Detectar inconsistencias o manipulaciones.
- Generar explicabilidad del sistema.

---

### Ejemplos de eventos

```text
ballot_scanned
vote_detected
inconsistency_detected
manual_override
results_submitted
dashboard_updated
```

---

### Información importante que puede almacenar

| Campo | Descripción |
|---|---|
| `id` | Identificador del evento |
| `entity_type` | Tipo de entidad relacionada |
| `entity_id` | ID de la entidad relacionada |
| `event_type` | Tipo de evento |
| `user_id` | Usuario responsable |
| `timestamp` | Fecha y hora del evento |
| `details` | Información adicional en JSON |
| `hash` | Hash para integridad |

---

# 6. `inconsistencies`

### ¿Qué es?
La entidad `inconsistencies` almacena todas las inconsistencias detectadas por el sistema.

Estas inconsistencias pueden ser:
- errores de captura,
- boletas ambiguas,
- diferencias de conteo,
- o conflictos entre validaciones.

---

### ¿Para qué sirve?
Permite:

- Registrar anomalías detectadas.
- Facilitar revisiones humanas.
- Generar alertas en tiempo real.
- Mejorar la transparencia del sistema.
- Mantener evidencia de correcciones.

---

### Información importante que puede almacenar

| Campo | Descripción |
|---|---|
| `id` | Identificador de inconsistencia |
| `ballot_id` | Boleta relacionada |
| `inconsistency_type` | Tipo de inconsistencia |
| `severity` | Nivel de severidad |
| `resolved` | Indica si fue resuelta |
| `resolution_notes` | Notas de resolución |
| `resolved_by` | Usuario que resolvió |
| `created_at` | Timestamp del registro |

---

# Conclusión

Estas entidades conforman la base de datos principal del sistema PREP con IA.

El diseño está orientado no solamente al almacenamiento de resultados, sino principalmente a:

- auditoría,
- trazabilidad,
- explicabilidad,
- transparencia,
- y reconstrucción completa de eventos.

La entidad más importante del sistema es `events`, ya que permite construir una arquitectura basada en eventos capaz de registrar y explicar cada operación realizada dentro del proceso electoral.

