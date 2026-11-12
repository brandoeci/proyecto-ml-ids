# Cronograma — proyecto final MLEA_M

sustentación: **viernes 20 de noviembre**, en el horario de clase. hoy es jueves 12, quedan 8 días.
equipo: hildebrando, isaac, jared, angélica.

---

## antes que nada: somos 4 y la guía dice máximo 3

sección 2 de la guía, textual: *"Puede trabajar en grupos de máximo 3 personas"*.
**hay que preguntarle al profesor esta semana, no el 19.** si no deja, toca
partirse en 3 + 1 o pedir excepción por escrito. el cronograma asume 4 y se
ajusta fácil si toca.

---

## quién lleva qué

repartido por peso en la rúbrica, no por número de secciones.

| | bloque | secciones | puntos | minutos |
|---|---|---|---|---|
| **angélica** | problema y datos | §1–§2 | 8 (problema 4 + datos 4) | 2:30 |
| **jared** | eda | §3 | 8 (eda) | 3:00 |
| **isaac** | modelos y métricas | §4–§5 | 20 (modelos 14 + métricas 6) | 3:30 |
| **hildebrando** | hiperparámetros, desconocidos, cascada y cierre | §6–§10 | 8 (hiperparám. 4 + aplicación 4) | 3:00 |

los 6 puntos que faltan (código 3 + presentación 3) son de los cuatro.

---

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

---

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

---

## riesgos

| riesgo | qué hacemos |
|---|---|
| somos 4 y la guía dice 3 | preguntar **esta semana**, no el 19 |
| nos pasamos de 12 minutos | ensayo del martes con cronómetro; lo primero que se recorta son las figuras del eda |
| preguntan de una sección que no es la mía | las dos rondas de preguntas cruzadas, sábado y miércoles |
| no se puede sustentar en vivo | video grabado el jueves 19 |
| a alguien no le corre el notebook | `requirements.txt`, y el jueves 12 todos lo corren |
| los grupos se escogen al azar | la guía lo dice; hay que estar listos desde el lunes, no desde el jueves |
