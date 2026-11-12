# Cronograma — Proyecto final MLEA_M

**Sustentación: viernes 20 de noviembre de 2026, en el horario de clase.**
Hoy es jueves 12. Quedan 8 días.

**Equipo:** Hildebrando, Isaac, Jared, Angélica.

---

## ⚠️ Antes que nada: somos 4 y la guía dice máximo 3

La guía del proyecto, sección 2, dice textualmente: *"Puede trabajar en grupos de
máximo 3 personas"*. **Hay que preguntarle al profesor esta semana**, no el 19.
Si no deja, las opciones son partirse en 3 + 1 o pedir excepción por escrito.
Mientras se resuelve, el cronograma asume 4 personas y el reparto se ajusta
fácil si toca.

---

## Dónde estamos hoy (12 de noviembre)

Lo que ya está hecho y corriendo:

- Descarga y muestreo del dataset (16.2 M de flujos → 1.42 M de trabajo)
- Consolidación, deduplicado y partición train/test
- Notebook completo con las 10 secciones, ejecutado de punta a punta
- 11 modelos entrenados: 5 en la etapa 1, 6 en la etapa 2
- Búsqueda de hiperparámetros, experimento de conjunto abierto y cascada completa

Lo que falta, y es el trabajo de estos 8 días:

- **Que cada uno entienda su sección de verdad**, no que la lea. El profesor
  puede preguntar cualquier cosa y los grupos se escogen al azar.
- Las diapositivas
- Ensayar con cronómetro hasta que quepa en 12 minutos
- Grabar el video de respaldo (la guía lo permite si no se puede sustentar)

---

## Día por día

| Día | Qué se hace | Quién | Qué queda entregado |
|---|---|---|---|
| **jue 12** | Clonar el repo, correr el notebook completo cada uno en su máquina. Confirmar que a todos les da lo mismo. Preguntarle al profe lo de los 4 integrantes. | todos | Todos con el entorno andando |
| **vie 13** | Repartirse las secciones (ver abajo). Cada uno lee su parte del notebook línea por línea y anota lo que no entiende. | todos | Lista de dudas por persona |
| **sáb 14** | Resolver las dudas entre todos. Es la sesión más importante de la semana: aquí se decide si el grupo puede defender el trabajo o no. | todos | Cero dudas abiertas |
| **dom 15** | Cada uno escribe el guion de su bloque: qué dice, en qué orden, con qué número cierra. | todos | 4 guiones escritos |
| **lun 16** | Diapositivas. Una plantilla común, cada uno llena sus láminas con las figuras de `reports/figuras/`. | todos | Borrador completo del deck |
| **mar 17** | Primer ensayo completo con cronómetro. Casi seguro se pasa de 12 min: recortar. | todos | Deck recortado, tiempo medido |
| **mié 18** | Segundo ensayo. Rondas de preguntas cruzadas entre nosotros: cada uno le pregunta al otro lo más incómodo de su sección. | todos | Respuestas preparadas |
| **jue 19** | Ensayo final. Grabar el video de 12 min como respaldo. Subir el notebook a la tarea de Teams. | todos | Video grabado + notebook entregado |
| **vie 20** | **Sustentación.** | todos | — |

---

## Reparto de la sustentación (12 minutos)

Repartido por peso en la rúbrica, no por número de secciones.

| # | Bloque | Secciones | Puntos de la rúbrica | Tiempo | Responsable |
|---|---|---|---|---|---|
| 1 | **Problema y datos** — qué se resuelve, por qué dos etapas, de dónde salen los datos y qué defectos tienen | §1, §2 | 8 pts (Problema 4 + Datos 4) | 2:30 | **Angélica** |
| 2 | **EDA** — los 5 insights, y sobre todo la fuga de `Dst Port` | §3 | 8 pts (EDA) | 3:00 | **Jared** |
| 3 | **Modelos y métricas** — los 5 de la etapa 1, los 6 de la etapa 2, por qué F1 macro y no accuracy | §4, §5 | 20 pts (Modelos 14 + Métricas 6) | 3:30 | **Isaac** |
| 4 | **Hiperparámetros, ataques desconocidos, cascada y cierre** | §6–§10 | 8 pts (Hiperparámetros 4 + Aplicación 4) | 3:00 | **Hildebrando** |

Total: 12:00. Los 6 puntos restantes (Código 3 + Presentación 3) son de todos.

**El reparto se puede cambiar**, pero que cada uno escoja su bloque el viernes 13
y no el miércoles 18.

---

## Lo que tiene que decir cada bloque (resumen de una línea)

1. **Angélica** — "Un SOC recibe millones de flujos al día y las firmas solo
   detectan lo ya catalogado. Y este dataset famoso tiene 10 defectos que
   encontramos."
2. **Jared** — "Dos columnas bastaban para clasificar casi todo. Era fuga, no
   señal. La quitamos y nuestras métricas bajaron: por eso son creíbles."
3. **Isaac** — "Entrenamos 11 modelos. El mejor sin etiquetas es Isolation
   Forest, el mejor con etiquetas es LightGBM, y la diferencia entre los dos es
   el precio exacto de no tener ataques etiquetados."
4. **Hildebrando** — "Un clasificador normal ante un ataque nuevo solo puede
   ignorarlo o bautizarlo mal. El nuestro puede decir que no sabe, y eso es lo
   que justifica las dos etapas."

---

## Preguntas que el profesor va a hacer (prepararlas)

Cada una tiene su respuesta en el notebook. Hay que saber **en qué sección está**.

| Pregunta probable | Dónde está la respuesta |
|---|---|
| ¿Por qué no usaron el dataset completo? | §2 — disco de 23 GB y archivo de 3.8 GB; se cuenta la distribución real antes de muestrear |
| ¿Por qué accuracy tan alta? ¿No hay sobreajuste? | §3.1 y §5.2 — por eso se decide con F1 macro; el 83% benigno hace la accuracy inútil |
| ¿Por qué quitaron `Dst Port`? | §3.5 — hay un experimento que lo mide: 2 columnas igualan a 48 |
| ¿Por qué no un split aleatorio? | §3.9 — 28.8% de duplicados exactos; al azar se inflan las métricas |
| ¿Por qué dos etapas y no un modelo? | §8.1 — está la comparación contra el monolítico de 7 clases |
| ¿Por qué falla `Infiltracion`? | §3.7, §3.8 y §10 — se superpone con el tráfico benigno; lo avisó el K-Means |
| ¿Cómo eligieron el umbral? | §5.1 (FPR 1% sobre calibración) y §7 (τ por cobertura del 95%) |
| ¿Qué pasa si llega un ataque nuevo? | §7 — experimento de dejar una familia fuera |

---

## Riesgos

| Riesgo | Qué hacer |
|---|---|
| Somos 4 y la guía dice 3 | Preguntar esta semana, no el 19 |
| Nos pasamos de 12 minutos | Ensayo del mar 17 con cronómetro; es lo primero que se recorta |
| Preguntan algo de una sección que no es la mía | Sesión de preguntas cruzadas del mié 18 |
| Se cae el internet / no se puede sustentar | Video grabado el jue 19 |
| Alguien no puede correr el notebook | `requirements.txt` + el jue 12 todos lo corren |
