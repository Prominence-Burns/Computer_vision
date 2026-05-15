# Especificación: Acta de Escrutinio y Cómputo de Casilla (AECC)

**Fundamento legal:** Arts. 288, 290, 291, 292 LGIPE · Acuerdo INE/CG598/2023  
**Versión:** Proceso Electoral Federal 2023-2024

---

## Propósito

Este documento define la estructura de salida esperada para un sistema de clasificación automática de boletas electorales. Cada boleta procesada produce un evento JSON que se acumula para construir el AECC final de la casilla.

El flujo es:

```
Boleta física → Clasificador → JSON de boleta → Agregador → JSON de casilla → AECC
```

---

## Parte 1 — JSON por boleta individual

Cada boleta clasificada debe emitir un objeto con la siguiente estructura.

### Schema

```json
{
  "$schema": "urn:ine:aecc:boleta:v1",
  "boleta_id": "string — folio único de la boleta",
  "casilla_id": "string — identificador de casilla",
  "timestamp": "string — ISO 8601, momento de clasificación",

  "clasificacion": {
    "tipo": "valido | nulo | no_registrado",
    "subtipo": "ver tabla de subtipos abajo",
    "destinatario": "string | null — id del partido, coalición o 'CNR'",
    "confianza": "number — 0.0 a 1.0, certeza del clasificador",
    "requiere_revision": "boolean — true si debe ir a revisión humana"
  },

  "marcas_detectadas": [
    {
      "recuadro": "string — id del recuadro (ver catálogo de recuadros)",
      "tipo_marca": "cruz | raya | palomita | circulo | texto | mancha | reflejo | otro",
      "intensidad": "clara | tenue | borrosa",
      "dentro_recuadro": "boolean",
      "proporcion_en_recuadro": "number — 0.0 a 1.0, fracción de la marca dentro del recuadro"
    }
  ],

  "anomalias": [
    {
      "tipo": "talón_adherido | rotura | mancha_sello | reflejo_doblez | ilegible",
      "descripcion": "string — descripción breve",
      "afecta_validez": "boolean"
    }
  ],

  "texto_detectado": "string | null — texto libre escrito por el elector, si existe",

  "fundamento_legal": "string — resolución TEPJF o artículo LGIPE que sustenta la clasificación"
}
```

### Tabla de subtipos de clasificación

| `tipo`          | `subtipo`                        | Descripción                                                              | Fundamento              |
|-----------------|----------------------------------|--------------------------------------------------------------------------|-------------------------|
| `valido`        | `marca_estandar`                 | Cruz o X clara dentro del recuadro                                       | Art. 288 LGIPE          |
| `valido`        | `marca_atipica`                  | Raya, palomita, figura geométrica dentro del recuadro                    | SUP-JIN-081/2006        |
| `valido`        | `marca_fuera_recuadro`           | Mayor proporción de la marca cae en un solo recuadro                     | SUP-JIN-021/2006        |
| `valido`        | `recuadro_encerrado`             | El elector encerró el recuadro con un círculo o trazo                    | SUP-JIN-005/2006        |
| `valido`        | `texto_no_ofensivo`              | Leyenda o texto sin contenido denostativo dentro del recuadro            | SUP-JIN-051/2012        |
| `valido`        | `multimarca_positiva`            | X clara en un recuadro + marcas de rechazo en los demás                  | SUP-JIN-011/2012        |
| `valido`        | `nominativo_nombre`              | Nombre completo del candidato escrito en el espacio correspondiente      | SUP-JIN-246/2006        |
| `valido`        | `nominativo_apodo`               | Apodo, sobrenombre o siglas de conocimiento público                      | INE/CG517/2018          |
| `valido`        | `coalicion_multimarca`           | Marcas en dos o más recuadros de partidos de la misma coalición          | Art. 288 pár. 3 LGIPE   |
| `valido`        | `marca_tenue_patron`             | Marca incompleta pero con patrón claro respecto a otra marca completa    | SUP-JIN-014/2012        |
| `nulo`          | `marca_total`                    | Cruces o rayas sobre toda o gran parte de la boleta                      | SM-JIN-046/2015         |
| `nulo`          | `insulto`                        | Expresión denostativa o injuriosa en el recuadro marcado                 | SUP-JIN-069/2006        |
| `nulo`          | `multimarca_no_coaligados`       | Marcas en dos o más partidos que no forman coalición                     | SUP-JIN-028/2012        |
| `nulo`          | `blanco`                         | Sin ninguna marca que permita inferir intención                          | SUP-JIN-081/2006        |
| `nulo`          | `rotura_grave`                   | Boleta cortada o mutilada que impide conocer la intención                | SUP-JIN-085/2006        |
| `nulo`          | `nominativo_contradictorio`      | Nombre del candidato + partido que no lo postuló                         | INE/CG517/2018          |
| `no_registrado` | `nombre_candidato_nr`            | Nombre escrito que no corresponde a ningún candidato registrado          | SUP-JIN-246/2006        |
| `no_registrado` | `siglas_nr`                      | Siglas o apodo sin partido con registro en esta elección                 | SM-JIN-046/2015         |

### Catálogo de recuadros (`recuadro`)

Los IDs de recuadro corresponden a la boleta del proceso. Para elecciones federales 2024:

| ID             | Partido / opción                                |
|----------------|-------------------------------------------------|
| `PAN`          | Partido Acción Nacional                         |
| `PRI`          | Partido Revolucionario Institucional            |
| `PRD`          | Partido de la Revolución Democrática            |
| `PVEM`         | Partido Verde Ecologista de México              |
| `PT`           | Partido del Trabajo                             |
| `MC`           | Movimiento Ciudadano                            |
| `MORENA`       | Morena                                          |
| `SHH`          | Coalición Sigamos Haciendo Historia (Morena+PT+PVEM) |
| `FCM`          | Coalición Fuerza y Corazón por México (PAN+PRI+PRD) |
| `CI`           | Candidatura Independiente                       |
| `CNR`          | Espacio candidatos no registrados               |

> **Regla de coalición:** si el elector marca recuadros de partidos dentro de la misma coalición, `destinatario` debe ser el ID de la coalición (`SHH` o `FCM`), no el partido individual. El voto se registra en el renglón de la coalición en el AECC.

### Ejemplos de JSON por boleta

**Voto válido estándar para Morena:**
```json
{
  "$schema": "urn:ine:aecc:boleta:v1",
  "boleta_id": "B-00134",
  "casilla_id": "NL-01-865-B",
  "timestamp": "2024-06-02T18:42:11Z",
  "clasificacion": {
    "tipo": "valido",
    "subtipo": "marca_estandar",
    "destinatario": "MORENA",
    "confianza": 0.97,
    "requiere_revision": false
  },
  "marcas_detectadas": [
    {
      "recuadro": "MORENA",
      "tipo_marca": "cruz",
      "intensidad": "clara",
      "dentro_recuadro": true,
      "proporcion_en_recuadro": 0.93
    }
  ],
  "anomalias": [],
  "texto_detectado": null,
  "fundamento_legal": "Art. 288 párrafo 3 LGIPE"
}
```

**Voto válido para coalición (elector marcó PT y PVEM, ambos de SHH):**
```json
{
  "$schema": "urn:ine:aecc:boleta:v1",
  "boleta_id": "B-00217",
  "casilla_id": "NL-01-865-B",
  "timestamp": "2024-06-02T18:55:03Z",
  "clasificacion": {
    "tipo": "valido",
    "subtipo": "coalicion_multimarca",
    "destinatario": "SHH",
    "confianza": 0.91,
    "requiere_revision": false
  },
  "marcas_detectadas": [
    {
      "recuadro": "PT",
      "tipo_marca": "cruz",
      "intensidad": "clara",
      "dentro_recuadro": true,
      "proporcion_en_recuadro": 0.88
    },
    {
      "recuadro": "PVEM",
      "tipo_marca": "cruz",
      "intensidad": "clara",
      "dentro_recuadro": true,
      "proporcion_en_recuadro": 0.85
    }
  ],
  "anomalias": [],
  "texto_detectado": null,
  "fundamento_legal": "Art. 288 párrafo 3 LGIPE — marcas en recuadros de partidos coaligados"
}
```

**Voto nulo — boleta en blanco:**
```json
{
  "$schema": "urn:ine:aecc:boleta:v1",
  "boleta_id": "B-00389",
  "casilla_id": "NL-01-865-B",
  "timestamp": "2024-06-02T19:10:44Z",
  "clasificacion": {
    "tipo": "nulo",
    "subtipo": "blanco",
    "destinatario": null,
    "confianza": 0.99,
    "requiere_revision": false
  },
  "marcas_detectadas": [],
  "anomalias": [],
  "texto_detectado": null,
  "fundamento_legal": "Art. 291 párrafo 1 inciso b) LGIPE — SUP-JIN-081/2006"
}
```

**Voto que requiere revisión humana — marca ambigua:**
```json
{
  "$schema": "urn:ine:aecc:boleta:v1",
  "boleta_id": "B-00512",
  "casilla_id": "NL-01-865-B",
  "timestamp": "2024-06-02T19:22:18Z",
  "clasificacion": {
    "tipo": "valido",
    "subtipo": "marca_fuera_recuadro",
    "destinatario": "MC",
    "confianza": 0.61,
    "requiere_revision": true
  },
  "marcas_detectadas": [
    {
      "recuadro": "MC",
      "tipo_marca": "cruz",
      "intensidad": "clara",
      "dentro_recuadro": false,
      "proporcion_en_recuadro": 0.58
    },
    {
      "recuadro": "PAN",
      "tipo_marca": "cruz",
      "intensidad": "tenue",
      "dentro_recuadro": false,
      "proporcion_en_recuadro": 0.21
    }
  ],
  "anomalias": [],
  "texto_detectado": null,
  "fundamento_legal": "SUP-JIN-021/2006 — mayor proporción en MC, pero confianza baja requiere revisión"
}
```

---

## Parte 2 — JSON de casilla (AECC agregado)

Una vez clasificadas todas las boletas, el agregador produce el JSON de casilla. Este objeto representa directamente el AECC.

### Schema

```json
{
  "$schema": "urn:ine:aecc:casilla:v1",

  "metadatos": {
    "casilla_id": "string",
    "entidad_federativa": "string",
    "municipio_o_delegacion": "string",
    "distrito": "string",
    "seccion": "string",
    "tipo_casilla": "basica | contigua | extraordinaria | especial",
    "tipo_eleccion": "presidencia | diputacion_mr | diputacion_rp | senado_mr | senado_rp | municipal",
    "proceso_electoral": "string — ej. '2023-2024'",
    "fecha_computo": "string — ISO 8601"
  },

  "bloque_1": {
    "boletas_recibidas": "integer — total entregadas a la casilla",
    "BS": "integer — boletas sobrantes (no usadas y canceladas)",
    "PV": "integer — personas que votaron en lista nominal",
    "RPPV": "integer — representantes PP/CI votaron fuera de lista",
    "SV": "integer — suma de votantes (PV + RPPV)",
    "BSU": "integer — boletas sacadas de la urna"
  },

  "bloque_2": {
    "resultados": [
      {
        "partido_o_coalicion": "string — nombre completo",
        "id": "string — ID del catálogo de recuadros",
        "es_coalicion": "boolean",
        "partidos_coalicion": "array<string> | null — IDs de partidos miembro, si es coalición",
        "votos": "integer"
      }
    ],
    "CNR": "integer — candidatos no registrados",
    "VN": "integer — votos nulos",
    "RV": "integer — resultado total de la votación (suma de todos los rubros)"
  },

  "consistencia": {
    "criterio_1_pv_rppv_sv": "boolean — PV + RPPV == SV",
    "criterio_2_sv_bsu": "boolean — SV == BSU",
    "criterio_3_bsu_rv": "boolean — BSU == RV",
    "criterio_4_sum_vi_rv": "boolean — suma de todos los rubros == RV",
    "acta_consistente": "boolean — true solo si los cuatro criterios son true",
    "tipo_error": "ninguno | numerico | llenado | numerico_y_llenado"
  },

  "incidentes": {
    "se_presentaron": "boolean",
    "descripcion": "string | null",
    "hojas_de_incidentes": "integer"
  },

  "boletas_procesadas": "integer — total de boletas clasificadas",
  "boletas_revision_humana": "integer — boletas que requirieron revisión manual",
  "hash_boletas": "string — hash SHA-256 del array de JSONs de boletas individuales"
}
```

### Ejemplo completo de JSON de casilla

```json
{
  "$schema": "urn:ine:aecc:casilla:v1",

  "metadatos": {
    "casilla_id": "NL-01-865-B",
    "entidad_federativa": "Nuevo León",
    "municipio_o_delegacion": "Monterrey",
    "distrito": "01",
    "seccion": "865",
    "tipo_casilla": "basica",
    "tipo_eleccion": "diputacion_mr",
    "proceso_electoral": "2023-2024",
    "fecha_computo": "2024-06-02T20:15:00Z"
  },

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
      {
        "partido_o_coalicion": "Morena",
        "id": "MORENA",
        "es_coalicion": false,
        "partidos_coalicion": null,
        "votos": 198
      },
      {
        "partido_o_coalicion": "Coalición Sigamos Haciendo Historia",
        "id": "SHH",
        "es_coalicion": true,
        "partidos_coalicion": ["MORENA", "PT", "PVEM"],
        "votos": 73
      },
      {
        "partido_o_coalicion": "Partido Acción Nacional",
        "id": "PAN",
        "es_coalicion": false,
        "partidos_coalicion": null,
        "votos": 95
      },
      {
        "partido_o_coalicion": "Coalición Fuerza y Corazón por México",
        "id": "FCM",
        "es_coalicion": true,
        "partidos_coalicion": ["PAN", "PRI", "PRD"],
        "votos": 49
      },
      {
        "partido_o_coalicion": "Movimiento Ciudadano",
        "id": "MC",
        "es_coalicion": false,
        "partidos_coalicion": null,
        "votos": 91
      },
      {
        "partido_o_coalicion": "Candidatura Independiente",
        "id": "CI",
        "es_coalicion": false,
        "partidos_coalicion": null,
        "votos": 12
      }
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

  "incidentes": {
    "se_presentaron": false,
    "descripcion": null,
    "hojas_de_incidentes": 0
  },

  "boletas_procesadas": 584,
  "boletas_revision_humana": 7,
  "hash_boletas": "a3f9c2d1e84b76f0..."
}
```

---

## Parte 3 — Reglas de negocio del clasificador

Estas reglas deben implementarse antes de emitir cualquier JSON de boleta.

### R1 — Prioridad de coalición sobre partido individual

Si se detectan marcas en dos o más recuadros y todos los recuadros marcados pertenecen a la misma coalición, `destinatario` es el ID de la coalición, no el partido. El voto se acumula en el renglón de la coalición en `bloque_2.resultados`.

```
si recuadros_marcados ⊆ partidos_de_coalicion(C)
  → tipo = "valido", subtipo = "coalicion_multimarca", destinatario = ID(C)
```

### R2 — Marca fuera de recuadro

Si `dentro_recuadro = false` pero `proporcion_en_recuadro >= 0.5` para un solo recuadro, el voto es válido para ese recuadro. Si la proporción más alta es menor a 0.5, `requiere_revision = true`.

```
si max(proporcion_en_recuadro) >= 0.5 y solo un recuadro con proporcion >= 0.5
  → tipo = "valido", subtipo = "marca_fuera_recuadro"
si max(proporcion_en_recuadro) < 0.5
  → requiere_revision = true, confianza baja
```

### R3 — Marcas de rechazo no invalidan el voto

Si un recuadro tiene `tipo_marca = "cruz"` o marca positiva, y otros recuadros contienen texto como "NO", "×", líneas onduladas interpretables como rechazo, el voto es válido para el recuadro con marca positiva.

```
si exactamente un recuadro tiene marca_positiva
  y otros recuadros tienen marcas_de_rechazo
  → tipo = "valido", subtipo = "multimarca_positiva"
```

### R4 — Texto: válido vs nulo

| Condición del texto                                   | Resultado      |
|-------------------------------------------------------|----------------|
| Texto no ofensivo en recuadro ya marcado              | Válido         |
| Texto de aprobación como única marca                  | Válido         |
| Insulto o expresión denostativa en recuadro marcado   | Nulo           |
| Nombre de candidato registrado sin marca gráfica      | Válido nominativo |
| Nombre de candidato + partido que no lo postuló       | Nulo           |
| Nombre desconocido sin partido con registro           | No registrado  |

### R5 — Umbral de confianza para revisión humana

| Rango de confianza | Acción                                              |
|--------------------|-----------------------------------------------------|
| `>= 0.85`          | Clasificación automática, `requiere_revision = false` |
| `0.60 – 0.84`      | Clasificación tentativa, `requiere_revision = true`  |
| `< 0.60`           | No clasificar, `requiere_revision = true`, tipo sin asignar |

### R6 — Consistencia del AECC

Al generar el JSON de casilla, verificar:

```
criterio_1: PV + RPPV == SV
criterio_2: SV == BSU
criterio_3: BSU == RV
criterio_4: sum(resultados[*].votos) + CNR + VN == RV

acta_consistente: criterio_1 AND criterio_2 AND criterio_3 AND criterio_4
```

Si `acta_consistente = false`, el sistema debe marcar el acta para revisión antes de su envío al sistema de cómputo distrital.

---

## Parte 4 — Diagrama de flujo de clasificación

```
Boleta
  │
  ├─ ¿Tiene alguna marca o texto?
  │     └─ No → tipo: nulo, subtipo: blanco
  │
  ├─ ¿Marca en toda la boleta o gran superficie?
  │     └─ Sí → tipo: nulo, subtipo: marca_total
  │
  ├─ ¿Contiene insulto o expresión denostativa en recuadro marcado?
  │     └─ Sí → tipo: nulo, subtipo: insulto
  │
  ├─ ¿Boleta rota que impide inferir intención?
  │     └─ Sí → tipo: nulo, subtipo: rotura_grave
  │
  ├─ ¿Solo texto en espacio CNR, sin marca en recuadros?
  │     ├─ Nombre de candidato registrado → tipo: valido, subtipo: nominativo_nombre
  │     └─ Nombre desconocido → tipo: no_registrado
  │
  ├─ ¿Marcas en recuadros?
  │     ├─ Un solo recuadro marcado → tipo: valido (aplicar R2 para posición)
  │     │
  │     ├─ Múltiples recuadros marcados
  │     │     ├─ Todos en la misma coalición → valido, subtipo: coalicion_multimarca, R1
  │     │     ├─ Un positivo + resto de rechazo → valido, subtipo: multimarca_positiva, R3
  │     │     └─ Dos o más positivos en partidos no coaligados → nulo, subtipo: multimarca_no_coaligados
  │     │
  │     └─ ¿Confianza < umbral? → requiere_revision = true
  │
  └─ Emitir JSON de boleta
```

---

## Notas de implementación

- El campo `hash_boletas` en el JSON de casilla debe calcularse sobre el array JSON serializado y ordenado por `boleta_id`, para garantizar la trazabilidad entre boletas individuales y el acta final.
- Los campos `PV` y `RPPV` en `bloque_1` no provienen del clasificador de boletas — son datos capturados por el funcionario de casilla (marcas "VOTÓ" en lista nominal). El clasificador solo produce `BSU` y `bloque_2`.
- El `RV` en `bloque_2` debe recalcularse siempre como `sum(resultados[*].votos) + CNR + VN` y compararse con `BSU` para el criterio III.
- Para boletas con `requiere_revision = true`, el sistema debe registrar tanto la clasificación tentativa del modelo como la clasificación final del revisor humano, conservando ambos valores para auditoría.
