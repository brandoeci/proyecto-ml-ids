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

# Cronograma

sustentación: **viernes 20 de noviembre**, en el horario de clase. hoy es jueves 12, quedan 8 días.
equipo: hildebrando, isaac, jared, angélica.

## antes que nada: somos 4 y la guía dice máximo 3

sección 2 de la guía, textual: *"Puede trabajar en grupos de máximo 3 personas"*.
**hay que preguntarle al profesor esta semana, no el 19.** si no deja, toca
partirse en 3 + 1 o pedir excepción por escrito. el cronograma asume 4 y se
ajusta fácil si toca.

## quién lleva qué

repartido por peso en la rúbrica, no por número de secciones.

| | bloque | secciones | puntos | minutos |
|---|---|---|---|---|
| **angélica** | problema y datos | §1–§2 | 8 (problema 4 + datos 4) | 2:30 |
| **jared** | eda | §3 | 8 (eda) | 3:00 |
| **isaac** | modelos y métricas | §4–§5 | 20 (modelos 14 + métricas 6) | 3:30 |
| **hildebrando** | hiperparámetros, desconocidos, cascada y cierre | §6–§10 | 8 (hiperparám. 4 + aplicación 4) | 3:00 |

los 6 puntos que faltan (código 3 + presentación 3) son de los cuatro.

## día a día

| día | hildebrando | isaac | jared | angélica |
|---|---|---|---|---|
| **jue 12 (hoy)** | repo en github, pipeline de datos andando y notebook corriendo entero. preguntarle al profe lo de los 4 integrantes | clonar y correr `ingest.py` + `preparar.py` esta noche. tiene que dar **1.420.028 filas**; si da otra cosa, avisar de una | clonar y dejar el entorno probado. abrir `reports/figuras/` y saber qué figura es cuál | clonar y dejar el entorno probado. leerse la guía del proyecto con la rúbrica al lado, que es la que nos califica |
| **vie 13** | §6–§10. el hito es explicar el experimento de dejar una familia fuera sin mirar el notebook | §4–§5. una frase por modelo: qué hace, qué ventaja, qué limitación. son 11 modelos | §3. tiene que poder defender la fuga de `Dst Port` con el número del experimento en la mano | §1–§2. los defectos del dataset de memoria, y por qué esto es clasificación **y** detección de anomalías a la vez |
| **sáb 14** | escribe las 5 preguntas más incómodas de su bloque y las contesta por escrito | lo mismo con el suyo: la peor es *"¿por qué tu accuracy es tan alta?"* | lo mismo. la peor: *"¿y si el profe dice que quitar `Dst Port` fue botar información?"* | lo mismo. la peor: *"¿por qué no usaron las 16 millones de filas?"* |

> **cierre del sábado, los cuatro: sesión de dudas cruzada.** cada uno responde las preguntas de los otros tres. nadie sale con dudas abiertas de su sección. esta es la sesión que decide si el grupo puede defender el trabajo o no.

| día | hildebrando | isaac | jared | angélica |
|---|---|---|---|---|
| **dom 15** | guion de §6–§10, cerrando con el número de la prueba de fuego | guion de §4–§5, cerrando con la brecha entre el no supervisado y el supervisado | guion de §3, cerrando con el experimento de `Dst Port` | guion de §1–§2, cerrando con los 10 defectos del dataset |
| **lun 16** | monta la plantilla del deck y llena sus láminas | sus láminas: tabla de los 11 modelos + matriz de confusión | sus láminas: los 6 insights del eda, una figura por insight | sus láminas: el diagrama de las dos etapas y la tabla de defectos |

> **cierre del lunes: deck completo en borrador.** aunque esté feo. de aquí en adelante solo se recorta y se pule.

| día | hildebrando | isaac | jared | angélica |
|---|---|---|---|---|
| **mar 17** | primer ensayo completo con cronómetro. casi seguro nos pasamos: él lleva el reloj y marca dónde se recorta | recorta su bloque, que es el más largo (3:30) | recorta. las figuras que no se expliquen en 20 segundos se van | recorta. la tabla de 10 defectos no se lee entera, se nombran 3 |
| **mié 18** | segunda ronda de preguntas cruzadas, ahora en frío y sin avisar quién responde | ídem | ídem | ídem |

> **cierre del miércoles: deck congelado.** nada nuevo entra después. de aquí solo se ensaya.

| día | hildebrando | isaac | jared | angélica |
|---|---|---|---|---|
| **jue 19** | ensayo final cronometrado. graba el video de respaldo de 12 min y lo deja montado | sube el notebook a la tarea de Teams y confirma que quedó | revisa que las figuras del deck sean las mismas que las del notebook entregado | lee la rúbrica punto por punto contra el deck y dice qué falta |
| **vie 20** | sustenta §6–§10 | sustenta §4–§5 | sustenta §3 | sustenta §1–§2 |

## las preguntas que va a hacer el profesor

cada una tiene respuesta en el notebook. lo importante es saber **en qué sección está**.

| pregunta | dónde está |
|---|---|
| ¿por qué no usaron el dataset completo? | §2 — disco de 23 gb y un archivo de 3.8 gb; se cuenta la distribución real antes de muestrear |
| ¿por qué esa accuracy tan alta? ¿no hay sobreajuste? | §3.1 y §5.2 — con 83% de tráfico benigno la accuracy no sirve; se decide con f1 macro |
| ¿por qué quitaron `Dst Port`? | §3.5 — hay un experimento que lo mide: 2 columnas igualan a 48 |
| ¿por qué no un split aleatorio? | §3.9 y §3.10 — 28,8% de duplicados exactos; al azar las métricas se inflan solas |
| ¿por qué dos etapas y no un modelo? | §8.1 — está la comparación contra el monolítico de 7 clases |
| ¿por qué falla `Infiltracion`? | §3.7, §3.8 y §10 — se superpone con el tráfico benigno; lo avisó el k-means antes de modelar |
| ¿cómo eligieron los umbrales? | §5.1 (fpr 1% sobre calibración) y §7 (τ por cobertura del 95%) |
| ¿y si llega un ataque nuevo? | §7 — el experimento de dejar una familia fuera |
| ¿por qué agruparon 14 etiquetas en 6 familias? | §3.2, y §3.9 lo refuerza: dos ataques quedaron en ~50 flujos tras deduplicar |
| ¿por qué ese escalado en la etapa 1 y no otro? | §4.1 — con transformación por cuantiles el one-class svm daba roc-auc 0,38, peor que el azar |

## riesgos

| riesgo | qué hacemos |
|---|---|
| somos 4 y la guía dice 3 | preguntar **esta semana**, no el 19 |
| nos pasamos de 12 minutos | ensayo del martes con cronómetro; lo primero que se recorta son las figuras del eda |
| preguntan de una sección que no es la mía | las dos rondas de preguntas cruzadas, sábado y miércoles |
| no se puede sustentar en vivo | video grabado el jueves 19 |
| a alguien no le corre el notebook | `requirements.txt`, y el jueves 12 todos lo corren |
| los grupos se escogen al azar | la guía lo dice; hay que estar listos desde el lunes, no desde el jueves |

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

---

## Resultados

Test: **208.314 flujos** (60.000 benignos + 148.314 ataques), partición temporal
dentro de cada (día, familia). Umbrales fijados sobre calibración, nunca sobre test.

### Etapa 1 — detección de anomalías

| Modelo | ROC-AUC | PR-AUC | Recall @ FPR 1% |
|---|---:|---:|---:|
| *Random Forest (referencia supervisada)* | *0.945* | *0.979* | ***77,0%*** |
| **Local Outlier Factor** ← el que va en la cascada | 0.778 | 0.901 | **22,2%** |
| Autoencoder | 0.826 | 0.901 | 7,4% |
| Isolation Forest | 0.621 | 0.778 | 5,4% |
| One-Class SVM | 0.410 | 0.690 | 7,2% |

La brecha entre **22,2% y 77,0%** es el número más importante del trabajo: es el
**precio de no tener ataques etiquetados**, medido y no estimado.

> El escalado resultó tan determinante como el modelo. Con `QuantileTransformer`
> ajustado solo con tráfico benigno, el One-Class SVM daba ROC-AUC **0,38 — peor
> que lanzar una moneda**, porque esa transformación recorta los valores fuera
> del rango normal, que es justo la señal a detectar. Con logaritmo con signo +
> estandarización, el Isolation Forest pasó de 0,538 a 0,621 de ROC-AUC y la
> cascada de 2,3% a 22,2% de recall.

### Etapa 2 — clasificación de la familia

| Modelo | Accuracy | **F1 macro** | Acc. balanceada |
|---|---:|---:|---:|
| **Random Forest + GridSearch** | — | **0.9765** | — |
| LightGBM | 0.9685 | 0.9688 | 0.9724 |
| XGBoost | 0.9696 | 0.9676 | 0.9677 |
| Random Forest | 0.9661 | 0.9573 | 0.9635 |
| k-NN | 0.9650 | 0.9511 | 0.9520 |
| MLP | 0.9435 | 0.8998 | 0.9061 |
| Regresión logística *(línea base)* | 0.9270 | 0.8280 | 0.9345 |

La distancia contra la regresión logística (0.977 vs 0.828 de F1 macro) es la
que justifica usar algo complejo.

### Ataques desconocidos — dejar una familia fuera

| Familia oculta | → `Desconocido` (acierto) | → nombre falso | Coste: rechazo indebido |
|---|---:|---:|---:|
| Fuerza bruta | **99,7%** | 0,3% | 14,3% |
| Botnet | **99,5%** | 0,5% | 16,2% |
| Infiltración | **98,6%** | 1,4% | 14,9% |
| DDoS | **96,1%** | 3,9% | 12,5% |
| DoS | **89,5%** | 10,5% | 8,5% |
| Ataque web | 60,5% | 39,5% | 14,2% |

Promedio: **90,6%** de los ataques nunca vistos se reconocen como desconocidos
en vez de recibir una etiqueta inventada.

### Sistema completo

| | |
|---|---|
| Alertas generadas | 33.557 (16,1% del tráfico) |
| Ataques detectados | 32.855 de 148.314 — **22,2%** |
| …y además con familia correcta | 31.451 — **21,2%** |
| Precisión de las alertas | **97,9%** son ataques reales |

### El hallazgo incómodo

Comparado contra un clasificador monolítico de 7 clases, ante `Botnet` oculta
a ambos:

| | Ignora como benigno | **Le pone un nombre FALSO** | Marca `Desconocido` |
|---|---:|---:|---:|
| Monolítico (7 clases) | 19,0% | **81,0%** | 0,0% |
| Cascada (2 etapas) | 99,5% | **0,5%** | 0,0% |

La cascada **no inventa nombres**, pero tampoco llega a decir `Desconocido`: la
etapa 1 no marca esos flujos, así que nunca alcanzan la etapa 2 donde vive el
mecanismo de rechazo. Ese mecanismo **sí funciona** —§7 lo demuestra al 99,5%—
pero el cuello de botella está antes.

**Conclusión del proyecto: nombrar el ataque es la parte fácil; detectarlo sin
etiquetas es la difícil.** Todo el trabajo futuro que vale la pena está en la
etapa 1, no en la 2.

> El monolítico detecta el 89,9% de los ataques, pero alertando sobre el **23,9%
> del tráfico benigno**. La cascada está fijada en 1% por diseño. Los dos números
> no son comparables tal cual: no están en el mismo punto de operación.

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
