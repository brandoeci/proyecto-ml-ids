# Detección de ataques de red en dos etapas

### ¿Es anómalo? → ¿Cómo se llama?

Proyecto final de **MLEA_M — Machine Learning**, 2026-2
Escuela Colombiana de Ingeniería Julio Garavito

**Equipo:** Hildebrando Peña · Isaac · Jared · Angélica

---

## La idea

Un detector de intrusiones que **primero decide si un flujo de red es anómalo**
—sin haber visto nunca un ataque— y **después le pone nombre** al ataque, con la
honestidad de responder `Desconocido` cuando no se parece a nada conocido.

### Por qué en dos etapas y no en una

Un clasificador único de 7 clases solo reconoce lo que ya vio. En seguridad eso
es justo el problema: **el ataque que importa es el que todavía no está en el
dataset**. Separar las etapas hace que cada una resuelva un problema distinto:

| | Etapa 1 | Etapa 2 |
|---|---|---|
| Pregunta | ¿esto es normal? | ¿qué ataque es? |
| Tipo de problema | detección de anomalías (**no supervisado**) | clasificación **multiclase** |
| Qué ve al entrenar | **solo tráfico benigno** | solo flujos de ataque |
| Qué aporta | detecta lo nunca visto | da un nombre accionable |

La etapa 1 entrenada únicamente con tráfico benigno es lo que hace realista el
montaje: una organización sí tiene registros de su tráfico normal, pero **no
tiene ataques etiquetados de su propia red**.

```
                      ┌──────────────────────┐
  flujo de red  ──▶   │ ETAPA 1              │
                      │ Isolation Forest     │
                      │ (solo vio benignos)  │
                      └──────────┬───────────┘
                                 │
                   ¿puntuación ≥ umbral(FPR 1%)?
                      │                      │
                     NO                     SÍ
                      ▼                      ▼
              ┌───────────────┐   ┌──────────────────────┐
              │  Sin alerta   │   │ ETAPA 2              │
              │  (benigno)    │   │ LightGBM + rechazo   │
              └───────────────┘   └──────────┬───────────┘
                                             │
                              ¿máx P(familia) ≥ τ?
                                  │                │
                                 SÍ               NO
                                  ▼                ▼
                         ┌────────────────┐  ┌──────────────┐
                         │ "DDoS", "Bot"… │  │ "Desconocido"│
                         └────────────────┘  └──────────────┘
```

---

## Los datos

**CSE-CIC-IDS2018** — Canadian Institute for Cybersecurity, University of New
Brunswick. Bucket público de AWS S3, sin credenciales.

| | |
|---|---|
| Flujos en el dataset completo | **16.232.943** |
| Variables predictoras | 79 (+ etiqueta) |
| Etiquetas originales | 15 (benigno + 14 ataques) |
| Familias tras agrupar | 7 (benigno + 6 familias) |
| Tráfico benigno | **83,07%** |
| Clase más pequeña | `SQL Injection`, **87 flujos** |
| Flujos usados para modelar | 1.420.028 (tras muestreo y deduplicado) |

### Cómo se maneja el tamaño

El dataset son 6.4 GB de CSV y la máquina tenía 23 GB libres. `src/ingest.py`
procesa los 10 archivos **uno a uno**: descarga, muestrea, guarda en Parquet y
**borra el CSV crudo** antes del siguiente. Pico de disco por debajo de 4.5 GB.

El muestreo no es uniforme y es a propósito: **todos los flujos de ataque se
conservan** (tope de 100.000 por etiqueta) y el **tráfico benigno se submuestrea
al 8%**. `SQL Injection` tiene 87 flujos en 16 millones — submuestrearla la
borraría del mapa.

La distribución **real y completa** se cuenta antes de muestrear y queda en
`data/processed/conteo_etiquetas_real.json`, para que el EDA muestre el
desbalance verdadero aunque el modelado use la muestra.

---

## Los 10 problemas que tiene este dataset

Encontrarlos fue parte del trabajo. CSE-CIC-IDS2018 es un *benchmark* muy citado
y tiene defectos conocidos:

| # | Problema | Qué se hizo |
|---|---|---|
| 1 | `Thuesday-20-02-2018` trae **4 columnas extra** que los otros 9 no tienen | se descartan: son identificadores de red, no comportamiento |
| 2 | Fila de encabezado **repetida en medio** de varios archivos | se filtran las filas con `Label == "Label"` |
| 3 | Celdas con el texto `"Infinity"` en `Flow Byts/s` y `Flow Pkts/s` | a `NaN`, imputadas con la mediana del train |
| 4 | **28,8% de duplicados exactos** | se eliminan; si no, el mismo flujo cae en train y test |
| 5 | Columnas **constantes** (varianza cero) | se identifican y se descartan |
| 6 | Pares de variables con **correlación > 0.98** | se poda una de cada par |
| 7 | **`Dst Port` es fuga de información** | se demuestra y se excluye del modelo final |
| 8 | **`Infilteration` es indistinguible** del tráfico benigno | se reporta como hallazgo, no se esconde |
| 9 | **Desbalance extremo** (83% benigno; 87 flujos la clase menor) | familias, métricas macro, `class_weight` |
| 10 | Etiqueta con **error tipográfico** (`Infilteration`) y escritura inconsistente | normalización antes de mapear |

### El hallazgo que más cambió el trabajo

`Dst Port` es el puerto de destino: 80 es web, 22 es SSH. Como los ataques del
experimento se dirigieron a servicios específicos, **el puerto casi identifica el
ataque por sí solo**. Un árbol con **2 columnas** (puerto y protocolo) alcanza una
exactitud comparable a uno con 48 variables de comportamiento.

Eso no significa que el puerto sea una buena variable: significa que en este
dataset **codifica qué experimento se estaba corriendo**. En una red real el
atacante elige el puerto. Se excluyó del modelo final — una decisión que *baja*
las métricas reportadas y las vuelve creíbles.

### El otro hallazgo: dos ataques son casi una sola fila repetida

El 28,8% de duplicados exactos es el número agregado, y esconde lo importante:
**la tasa no está repartida de forma pareja entre los ataques, ni de lejos.**

| Etiqueta | En la muestra | Tras deduplicar | % duplicados |
|---|---:|---:|---:|
| `FTP-BruteForce` | 100.000 | **53** | **99,95%** |
| `DoS attacks-SlowHTTPTest` | 100.000 | **54** | **99,95%** |
| `DDOS attack-HOIC` | 100.000 | 46.396 | 53,60% |
| `SSH-Bruteforce` | 100.000 | 50.199 | 49,80% |
| `Bot` | 100.000 | 50.955 | 49,05% |
| `DoS attacks-Hulk` | 100.000 | 52.432 | 47,57% |
| `Benign` | 1.077.982 | 925.666 | 14,13% |
| `Infilteration` | 161.934 | 140.610 | 13,17% |
| `DoS attacks-Slowloris` | 10.990 | 9.712 | 11,63% |
| `Brute Force -Web` | 611 | 553 | 9,49% |
| `SQL Injection` | 87 | 84 | 3,45% |
| `Brute Force -XSS` | 230 | 228 | 0,87% |
| `DoS attacks-GoldenEye` | 41.508 | 41.392 | 0,28% |
| `DDoS attacks-LOIC-HTTP` | 100.000 | 99.964 | 0,04% |
| `DDOS attack-LOIC-UDP` | 1.730 | 1.730 | **0,00%** |

Va de **0,00% a 99,95%**. No es un defecto parejo del dataset: es específico de
ciertos ataques. Tiene sentido — una fuerza bruta de FTP repite exactamente la
misma operación miles de veces y CICFlowMeter produce registros idénticos.

Pero la consecuencia es seria: **`FTP-BruteForce` y `DoS attacks-SlowHTTPTest`
dejan de existir como material de entrenamiento utilizable.** De 100.000 flujos
cada uno quedan 53 y 54.

Un trabajo que no deduplique nunca ve este problema — al contrario, obtiene
métricas *excelentes* en esas dos clases, porque está evaluando sobre copias
exactas de lo que memorizó.

**Aquí se nota por qué agrupar en familias fue la decisión correcta:**
`Fuerza bruta` sobrevive gracias a `SSH-Bruteforce` (50.199 flujos únicos) y
`DoS` sobrevive gracias a `Hulk` y `GoldenEye`. Con las 14 etiquetas originales,
dos clases habrían quedado con ~50 ejemplos.

> Las columnas "en la muestra" ya tienen aplicado el tope de 100.000 flujos por
> etiqueta de la ingesta, así que los porcentajes están medidos sobre nuestra
> muestra de trabajo y no sobre los 16,2 M completos. La tabla se regenera con
> `python -I src/preparar.py` y queda en `data/processed/metadatos_modelado.json`.

---

## Los modelos

**11 modelos entrenados**, con diversidad de enfoques deliberada.

### Etapa 1 — detección de anomalías (entrenada solo con tráfico benigno)

| Modelo | Enfoque | Limitación |
|---|---|---|
| Isolation Forest | aislamiento por cortes aleatorios | pierde anomalías locales |
| One-Class SVM | frontera en espacio de kernel | coste ~O(n²), obliga a submuestrear |
| Local Outlier Factor | densidad local vs. vecinos | caro en predicción |
| Autoencoder (MLP) | error de reconstrucción | más frágil de ajustar |
| *Random Forest* | *supervisado — techo de referencia* | *necesita ataques etiquetados* |

### Etapa 2 — clasificación de la familia

Random Forest (*bagging*) · LightGBM (*boosting*) · XGBoost (*boosting*) ·
k-NN (instancias) · MLP (red neuronal) · Regresión logística (lineal, línea base)

### Cómo se evalúa

- **Etapa 1:** ROC-AUC, **PR-AUC** (la que manda con desbalance) y
  **recall a FPR = 1%** — el presupuesto de falsas alarmas que un SOC tolera.
- **Etapa 2:** **F1 macro** como criterio de decisión, más accuracy balanceada,
  F1 ponderado y recall por familia. Nunca accuracy sola: con 83% de tráfico
  benigno, "todo normal" da 83% de accuracy y cero ataques detectados.
- Umbrales fijados sobre un **conjunto de calibración**, nunca sobre el test.

### Hiperparámetros

Tres técnicas, cada una donde corresponde:

- **RandomizedSearchCV** en LightGBM — 7 dimensiones, varias continuas
- **GridSearchCV** en Random Forest — pocos parámetros discretos
- **Rejilla manual + ROC-AUC** en Isolation Forest — no hay `y` al entrenar, así
  que `GridSearchCV` no aplica; se evalúa sobre calibración

---

## El experimento que justifica las dos etapas

Para cada una de las 6 familias: se **borra del entrenamiento por completo**, se
entrena con las 5 restantes, y se le presentan los flujos de la familia oculta.

- **✓ acierto** → la manda a `Desconocido`
- **✗ error peligroso** → le pone el nombre de otra familia

Luego se compara contra un **clasificador monolítico de 7 clases**. El monolítico
gana en los ataques que ya conoce. Pero ante uno nuevo **solo tiene dos salidas y
las dos son malas**: ignorarlo como benigno o ponerle un nombre falso. La cascada
tiene una tercera, que es la honesta.

Los números están en §7 y §8.1 del notebook.

---

## Estructura

```
proyecto-ml-ids/
├── notebooks/
│   ├── proyecto_ids.ipynb      # EL ENTREGABLE — 10 secciones, ejecutado
│   └── proyecto_ids.py         # el mismo notebook en formato percent (jupytext)
├── src/
│   ├── config.py               # todos los parámetros, en un solo sitio
│   ├── ingest.py               # descarga de S3, muestreo, borra el crudo
│   ├── preparar.py             # une, mapea familias, deduplica, parte train/test
│   └── ejecutar_notebook.py    # corre el notebook de punta a punta
├── data/                       # (no versionado — lo reconstruye ingest.py)
├── reports/figuras/            # las figuras que se generan al ejecutar
├── CRONOGRAMA.md               # reparto del trabajo y de la sustentación
└── requirements.txt
```

---

## Cómo reproducirlo

Requiere Python 3.12 y el [AWS CLI v2](https://aws.amazon.com/cli/) (el bucket es
público, no hacen falta credenciales).

```bash
pip install -r requirements.txt

python -I src/ingest.py       # ~15 min: baja los 10 CSV, muestrea, borra el crudo
python -I src/preparar.py     # ~3 min: une, deduplica y parte train/test
python src/ejecutar_notebook.py
```

O abrir el notebook a mano:

```bash
jupyter lab notebooks/proyecto_ids.ipynb
```

**Reproducibilidad:** semilla fija (`SEMILLA = 42`) en ingesta, muestreo,
particiones y modelos. Los topes de tamaño están todos en `src/config.py`.

Los modelos basados en distancias y kernels se entrenan sobre submuestras por
coste computacional (One-Class SVM 20.000, LOF 50.000, k-NN y MLP 40.000). Está
declarado en `src/config.py` y en §4 del notebook, no escondido.

---

## Limitaciones

Dicho sin maquillaje:

1. **El dataset es de 2018 y de una red sintética en AWS.** La etapa 1 tiene que
   reentrenarse con el tráfico benigno de la red donde se instale; no se
   trasplanta.
2. **`Infiltracion` no funciona.** Hay ataques cuyo comportamiento a nivel de
   flujo es indistinguible del tráfico normal. Para esos hace falta otra fuente
   de datos.
3. **Una sola partición temporal**, sin validación cruzada repetida en las
   métricas finales: no hay intervalos de confianza sobre las diferencias.
4. **`Ataque web` tiene 865 flujos en total.** Cualquier métrica sobre esa
   familia tiene incertidumbre grande.
5. **Un solo dataset.** No se sabe si lo aprendido transfiere a otra red.
6. Ninguna métrica aquí mide resistencia a **ataques adversariales**.

---

## Fuente de los datos

Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani.
*Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization*. ICISSP 2018.
<https://www.unb.ca/cic/datasets/ids-2018.html>
