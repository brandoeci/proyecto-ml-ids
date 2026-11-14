# Detección de ataques de red en dos etapas

Proyecto final de Machine Learning (MLEA_M), 2026-2.
Escuela Colombiana de Ingeniería Julio Garavito.

Equipo: Hildebrando Peña, Isaac, Jared Farfán, Angélica.

## Qué hace

Detectar ataques en tráfico de red, pero en dos pasos separados:

1. Primero decidir si un flujo de red es raro. Este paso se entrena solo con
   tráfico normal, sin mostrarle ni un ataque.
2. Si salió raro, el segundo paso le pone nombre a la familia del ataque (DoS,
   DDoS, botnet, etc.). Si no se parece a nada de lo que vio, responde
   "Desconocido" en vez de inventar un nombre.

## Por qué dos etapas

Al principio pensamos en un solo clasificador de 7 clases, pero tiene un
problema: solo reconoce lo que ya vio entrenando. Y en seguridad el ataque que
más importa es justamente el que todavía no está en ningún dataset.

Separando las etapas, cada una resuelve algo distinto:

| | Etapa 1 | Etapa 2 |
|---|---|---|
| Qué pregunta | ¿esto es normal? | ¿qué ataque es? |
| Tipo de aprendizaje | no supervisado | supervisado, multiclase |
| Con qué se entrena | solo tráfico benigno | solo flujos de ataque |

Lo otro que nos gustó es que se parece más a la realidad. Una empresa sí tiene
registros de su tráfico normal, pero no tiene ataques etiquetados de su propia
red.

El recorrido completo de un flujo es: entra a la etapa 1, que le calcula una
puntuación de anomalía. Si la puntuación queda por debajo del umbral, se
descarta como benigno. Si lo pasa, va a la etapa 2, que mira cuál es la
probabilidad más alta entre las 6 familias. Si esa probabilidad supera el umbral
de rechazo, reporta la familia; si no, reporta "Desconocido".

## Los datos

Usamos CSE-CIC-IDS2018, del Canadian Institute for Cybersecurity de la
Universidad de New Brunswick. Está en un bucket público de S3, se baja sin
credenciales.

| | |
|---|---|
| Flujos en el dataset completo | 16.232.943 |
| Variables | 79 más la etiqueta |
| Etiquetas originales | 15 (benigno y 14 ataques) |
| Familias después de agrupar | 7 (benigno y 6 familias) |
| Tráfico benigno | 83,07% |
| Clase más pequeña | SQL Injection, 87 flujos |
| Flujos que usamos para modelar | 1.420.028 |

### El problema del tamaño

Son 6.4 GB de CSV y el portátil tenía 23 GB libres, con un solo archivo que pesa
3.8 GB. Por eso `src/ingest.py` va archivo por archivo: baja uno, lo muestrea, lo
guarda en parquet y borra el CSV antes de seguir con el siguiente. El pico de
disco no pasó de 4.5 GB.

El muestreo no es parejo, y es a propósito. Nos quedamos con todos los flujos de
ataque (con tope de 100.000 por etiqueta) y bajamos el tráfico benigno al 8%.
SQL Injection tiene 87 flujos entre 16 millones: submuestreándola igual que el
resto, desaparecía.

Para que el EDA no quedara mintiendo, contamos la distribución real antes de
muestrear y la guardamos en `data/processed/conteo_etiquetas_real.json`. Las
gráficas del desbalance usan esos números y no los de la muestra.

## Problemas que tiene el dataset

Esto fue buena parte del trabajo. CSE-CIC-IDS2018 se cita mucho pero viene con
varios defectos. Algunos ya están reportados en la literatura y otros nos
salieron al abrirlo:

| # | Problema | Qué hicimos |
|---|---|---|
| 1 | `Thuesday-20-02-2018` trae 4 columnas que los otros 9 archivos no tienen | las descartamos, son identificadores de red y no comportamiento |
| 2 | La fila de encabezado aparece repetida en medio de varios archivos | filtramos las filas donde `Label == "Label"` |
| 3 | Celdas con el texto `"Infinity"` en `Flow Byts/s` y `Flow Pkts/s` | las pasamos a `NaN` e imputamos con la mediana del train |
| 4 | 28,8% de duplicados exactos | los quitamos, si no el mismo flujo cae en train y en test |
| 5 | Columnas constantes, con varianza cero | las identificamos y las sacamos |
| 6 | Pares de variables con correlación mayor a 0.98 | dejamos una de cada par |
| 7 | `Dst Port` es fuga de información | lo medimos y lo excluimos del modelo final |
| 8 | `Infilteration` es indistinguible del tráfico benigno | lo reportamos como hallazgo, no lo escondimos |
| 9 | Desbalance extremo (83% benigno, 87 flujos la clase menor) | familias, métricas macro y `class_weight` |
| 10 | Una etiqueta viene con error de ortografía (`Infilteration`) y la escritura es inconsistente | normalizamos antes de mapear |

### Dst Port se estaba llevando el crédito

`Dst Port` es el puerto de destino: 80 es web, 22 es SSH. Como los ataques del
experimento se lanzaron contra servicios específicos, el puerto casi identifica
el ataque por sí solo.

Lo medimos: un árbol entrenado únicamente con el puerto y el protocolo (2
columnas) llega a una exactitud parecida a uno con las 48 variables de
comportamiento. Eso no significa que el puerto sea una variable buena, significa
que en este dataset lo que codifica es qué experimento estaban corriendo. En una
red real el atacante escoge el puerto que quiera.

Lo sacamos del modelo final. Las métricas bajan, pero son las que se pueden
sostener.

### Dos ataques son casi la misma fila repetida

El 28,8% de duplicados es el promedio, y esconde lo interesante: la tasa no está
repartida parejo entre los ataques.

| Etiqueta | En la muestra | Tras deduplicar | % duplicados |
|---|---:|---:|---:|
| `FTP-BruteForce` | 100.000 | 53 | 99,95% |
| `DoS attacks-SlowHTTPTest` | 100.000 | 54 | 99,95% |
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
| `DDOS attack-LOIC-UDP` | 1.730 | 1.730 | 0,00% |

Va de 0,00% a 99,95%. Tiene lógica: una fuerza bruta de FTP repite exactamente
la misma operación miles de veces y CICFlowMeter saca registros idénticos.

El problema es que `FTP-BruteForce` y `DoS attacks-SlowHTTPTest` se quedan sin
material de entrenamiento. De 100.000 flujos cada uno quedan 53 y 54.

Un trabajo que no deduplique no ve este problema. Al contrario, le salen métricas
buenísimas en esas dos clases, porque está evaluando sobre copias exactas de lo
que el modelo memorizó.

Esto fue lo que nos convenció de agrupar las 14 etiquetas en 6 familias. "Fuerza
bruta" sobrevive por SSH-Bruteforce (50.199 flujos únicos) y "DoS" sobrevive por
Hulk y GoldenEye. Con las etiquetas originales nos quedaban dos clases con unos
50 ejemplos.

Las columnas "en la muestra" ya tienen aplicado el tope de 100.000 flujos por
etiqueta, así que los porcentajes están medidos sobre nuestra muestra y no sobre
los 16,2 millones completos. La tabla se regenera con `python -I src/preparar.py`
y queda en `data/processed/metadatos_modelado.json`.

## Los modelos

Entrenamos 11 modelos, buscando que fueran de enfoques distintos y no 11
variantes de lo mismo.

### Etapa 1, detección de anomalías

Todos entrenados solo con tráfico benigno.

| Modelo | Idea | Limitación |
|---|---|---|
| Isolation Forest | aísla puntos con cortes aleatorios | se le escapan las anomalías locales |
| One-Class SVM | arma una frontera alrededor de lo normal | cuesta cerca de O(n²), obliga a submuestrear |
| Local Outlier Factor | compara la densidad de cada punto con la de sus vecinos | caro al momento de predecir |
| Autoencoder (MLP) | mide el error al reconstruir la fila | más difícil de ajustar |
| Random Forest | supervisado, va como techo de referencia | necesita ataques etiquetados |

El Random Forest no hace parte del sistema. Lo metimos para poder medir cuánto
se pierde por no tener etiquetas.

### Etapa 2, familia del ataque

Random Forest (bagging), LightGBM y XGBoost (boosting), k-NN (por instancias),
MLP (red neuronal) y regresión logística como línea base.

### Cómo los comparamos

En la etapa 1: ROC-AUC, PR-AUC y recall con el FPR fijado en 1%. El PR-AUC es el
que manda porque hay mucho desbalance, y el 1% de falsas alarmas es un
presupuesto realista para un analista.

En la etapa 2 decidimos con F1 macro, y además reportamos accuracy balanceada,
F1 ponderado y recall por familia. Accuracy sola no sirve: con 83% de tráfico
benigno, un modelo que diga "todo normal" saca 83% y detecta cero ataques.

Los dos umbrales se fijan sobre un conjunto de calibración aparte. El test no se
toca para nada.

### Hiperparámetros

Usamos tres técnicas, cada una donde tenía sentido:

- RandomizedSearchCV en LightGBM, porque tiene 7 parámetros y varios son
  continuos. Probar la rejilla completa no es viable.
- GridSearchCV en Random Forest, que son pocos parámetros y discretos.
- Rejilla a mano en Isolation Forest. Acá no hay `y` al entrenar, así que
  GridSearchCV no aplica porque no puede calcular un score; evaluamos cada
  combinación sobre calibración.

## El experimento de los ataques desconocidos

Para probar si el sistema sabe decir "no sé", hicimos lo mismo con cada una de
las 6 familias: la borramos completa del entrenamiento, entrenamos con las 5 que
quedan y después le mostramos los flujos de la familia escondida.

Si la manda a "Desconocido", acertó. Si le pone el nombre de otra familia, ese es
el error peligroso, porque el analista sale a atender lo que no es.

Después comparamos contra un clasificador monolítico de 7 clases. El monolítico
gana en los ataques que ya conoce, pero ante uno nuevo solo tiene dos salidas y
las dos son malas: ignorarlo como benigno o ponerle un nombre falso.

## Resultados

El test son 208.314 flujos (60.000 benignos y 148.314 ataques), con partición
temporal dentro de cada día y familia.

### Etapa 1

| Modelo | ROC-AUC | PR-AUC | Recall con FPR al 1% |
|---|---:|---:|---:|
| Random Forest (referencia supervisada) | 0.945 | 0.979 | 77,0% |
| Local Outlier Factor (el que quedó en la cascada) | 0.778 | 0.901 | 22,2% |
| Autoencoder | 0.826 | 0.901 | 7,4% |
| Isolation Forest | 0.621 | 0.778 | 5,4% |
| One-Class SVM | 0.410 | 0.690 | 7,2% |

La diferencia entre 22,2% y 77,0% es el número más importante del trabajo: es lo
que cuesta no tener ataques etiquetados, medido y no estimado.

Algo que no esperábamos es que el escalado pesara tanto como el modelo. Con
`QuantileTransformer` ajustado solo con tráfico benigno, el One-Class SVM daba
ROC-AUC de 0,38, peor que lanzar una moneda. La razón es que esa transformación
recorta los valores que caen fuera del rango normal, y eso es justo la señal que
queremos detectar. Cambiándolo por logaritmo con signo y estandarización, el
Isolation Forest subió de 0,538 a 0,621 de ROC-AUC y la cascada completa pasó de
2,3% a 22,2% de recall.

### Etapa 2

| Modelo | Accuracy | F1 macro | Acc. balanceada |
|---|---:|---:|---:|
| Random Forest con GridSearch | | 0.9765 | |
| LightGBM | 0.9685 | 0.9688 | 0.9724 |
| XGBoost | 0.9696 | 0.9676 | 0.9677 |
| Random Forest | 0.9661 | 0.9573 | 0.9635 |
| k-NN | 0.9650 | 0.9511 | 0.9520 |
| MLP | 0.9435 | 0.8998 | 0.9061 |
| Regresión logística (línea base) | 0.9270 | 0.8280 | 0.9345 |

La distancia contra la regresión logística (0.977 frente a 0.828 de F1 macro) es
lo que justifica usar algo más complejo. Si la línea base hubiera quedado cerca,
no tendría sentido.

### Dejando una familia fuera

| Familia escondida | Va a "Desconocido" (acierto) | Recibe nombre falso | Rechazo indebido |
|---|---:|---:|---:|
| Fuerza bruta | 99,7% | 0,3% | 14,3% |
| Botnet | 99,5% | 0,5% | 16,2% |
| Infiltración | 98,6% | 1,4% | 14,9% |
| DDoS | 96,1% | 3,9% | 12,5% |
| DoS | 89,5% | 10,5% | 8,5% |
| Ataque web | 60,5% | 39,5% | 14,2% |

En promedio, el 90,6% de los ataques que nunca vio los reconoce como
desconocidos en vez de ponerles una etiqueta inventada.

### El sistema completo

| | |
|---|---|
| Alertas generadas | 33.557, que es el 16,1% del tráfico |
| Ataques detectados | 32.855 de 148.314, el 22,2% |
| Detectados y además con la familia correcta | 31.451, el 21,2% |
| Precisión de las alertas | 97,9% son ataques reales |

### El resultado incómodo

Comparando contra el monolítico de 7 clases, con Botnet escondida para los dos:

| | Lo ignora como benigno | Le pone nombre FALSO | Dice "Desconocido" |
|---|---:|---:|---:|
| Monolítico (7 clases) | 19,0% | 81,0% | 0,0% |
| Cascada (2 etapas) | 99,5% | 0,5% | 0,0% |

La cascada no inventa nombres, pero tampoco alcanza a decir "Desconocido": la
etapa 1 no marca esos flujos, así que nunca llegan a la etapa 2, que es donde
vive el mecanismo de rechazo. Ese mecanismo sí funciona, el experimento de dejar
una familia fuera lo muestra al 99,5%, pero el cuello de botella está antes.

La conclusión del proyecto es esa: ponerle nombre al ataque es la parte fácil,
detectarlo sin etiquetas es la difícil. Lo que valdría la pena mejorar está en la
etapa 1, no en la 2.

Hay que decir también que el monolítico detecta el 89,9% de los ataques, pero
alertando sobre el 23,9% del tráfico benigno. La cascada está fijada en 1% por
diseño. Los dos números no son comparables así nomás, porque no están en el mismo
punto de operación.

## Lo que no quedó resuelto

1. El dataset es de 2018 y de una red sintética montada en AWS. La etapa 1 hay
   que reentrenarla con el tráfico normal de la red donde se vaya a instalar, no
   se puede trasplantar así.
2. `Infiltracion` no funciona. Hay ataques cuyo comportamiento a nivel de flujo
   es indistinguible del tráfico normal, y para esos haría falta otra fuente de
   datos, por ejemplo logs del host.
3. Hicimos una sola partición temporal, sin validación cruzada repetida en las
   métricas finales. Eso significa que no tenemos intervalos de confianza, así
   que no podemos afirmar que las diferencias pequeñas entre modelos sean reales.
4. `Ataque web` tiene 865 flujos en total. Cualquier métrica sobre esa familia
   tiene mucha incertidumbre.
5. Es un solo dataset, así que no sabemos si lo que aprendimos sirve en otra red.
6. Ninguna de las métricas mide qué pasa si el atacante sabe que hay un modelo y
   trata de evadirlo.

## Estructura del repo

```
proyecto-ml-ids/
├── notebooks/
│   ├── proyecto_ids.ipynb      # el entregable, 10 secciones
│   └── proyecto_ids.py         # el mismo notebook en formato percent (jupytext)
├── src/
│   ├── config.py               # los parámetros, en un solo sitio
│   ├── ingest.py               # descarga de S3, muestreo, borra el crudo
│   └── preparar.py             # une, mapea familias, deduplica, parte train/test
├── data/                       # no va en git, lo reconstruye ingest.py
├── reports/figuras/            # las figuras que salen al ejecutar el notebook
└── requirements.txt
```

## Cómo correrlo

Necesita Python 3.12 y el [AWS CLI v2](https://aws.amazon.com/cli/). El bucket es
público, no hay que configurar credenciales.

```bash
pip install -r requirements.txt

python -I src/ingest.py       # ~15 min: baja los 10 CSV, muestrea y borra el crudo
python -I src/preparar.py     # ~3 min: une, deduplica y parte train/test
jupyter lab notebooks/proyecto_ids.ipynb
```

El notebook corre entero de arriba a abajo en unos 23 minutos y pide cerca de
2,5 GB de RAM.

Todo lleva semilla fija (`SEMILLA = 42`) en la ingesta, el muestreo, las
particiones y los modelos. Los topes de tamaño están en `src/config.py`.

Los modelos que dependen de distancias o kernels se entrenan sobre submuestras
porque si no, no terminan: One-Class SVM con 20.000 filas, LOF con 50.000, k-NN y
MLP con 40.000. Está puesto en `src/config.py` y dicho en la sección 4 del
notebook.

# Cronograma

sustentación: **viernes 20 de noviembre**, en el horario de clase.
equipo: hildebrando, isaac, jared, angélica.

## lo de ser 4: resuelto

la guía dice *"grupos de máximo 3 personas"*, pero **el profesor nos autorizó
los 4**. el reparto de abajo queda firme.

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
| **jue 12** | repo en github, pipeline de datos andando y notebook corriendo entero. confirmado con el profe que podemos ser 4 | clonar y correr `ingest.py` + `preparar.py`. tiene que dar **1.420.028 filas**; si da otra cosa, avisar de una | clonar y dejar el entorno probado. abrir `reports/figuras/` y saber qué figura es cuál | clonar y dejar el entorno probado. leerse la guía del proyecto con la rúbrica al lado, que es la que nos califica |
| **vie 13** | §6–§10. el hito es explicar el experimento de dejar una familia fuera sin mirar el notebook | §4–§5. una frase por modelo: qué hace, qué ventaja, qué limitación. son 11 modelos | §3. tiene que poder defender la fuga de `Dst Port` con el número del experimento en la mano | §1–§2. los defectos del dataset de memoria, y por qué esto es clasificación **y** detección de anomalías a la vez |
| **sáb 14** | escribe las 5 preguntas más incómodas de su bloque y las contesta por escrito | lo mismo con el suyo: la peor es *"¿por qué tu accuracy es tan alta?"* | lo mismo. la peor: *"¿y si el profe dice que quitar `Dst Port` fue botar información?"* | lo mismo. la peor: *"¿por qué no usaron las 16 millones de filas?"* |

> **cierre del sábado, los cuatro: sesión de dudas cruzada.** cada uno responde las preguntas de los otros tres. nadie sale con dudas abiertas de su sección. esta es la sesión que decide si el grupo puede defender el trabajo o no. cada uno llena su archivo en [`preguntas/`](preguntas/) y lo commitea desde su cuenta.

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
| ¿cómo eligieron los umbrales? | §5.1 (fpr 1% sobre calibración) y §7 (umbral de rechazo por cobertura del 95%) |
| ¿y si llega un ataque nuevo? | §7 — el experimento de dejar una familia fuera |
| ¿por qué agruparon 14 etiquetas en 6 familias? | §3.2, y §3.9 lo refuerza: dos ataques quedaron en ~50 flujos tras deduplicar |
| ¿por qué ese escalado en la etapa 1 y no otro? | §4.1 — con transformación por cuantiles el one-class svm daba roc-auc 0,38, peor que el azar |

## riesgos

| riesgo | qué hacemos |
|---|---|
| nos pasamos de 12 minutos | ensayo del martes con cronómetro; lo primero que se recorta son las figuras del eda |
| preguntan de una sección que no es la mía | las dos rondas de preguntas cruzadas, sábado y miércoles |
| no se puede sustentar en vivo | video grabado el jueves 19 |
| a alguien no le corre el notebook | `requirements.txt`, y todos lo corren el fin de semana |
| los grupos se escogen al azar | la guía lo dice; hay que estar listos desde el lunes, no desde el jueves |

## De dónde salieron los datos

Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani.
*Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization*. ICISSP 2018.
<https://www.unb.ca/cic/datasets/ids-2018.html>
