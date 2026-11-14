# preguntas de isaac (modelos y metricas)

secciones que sustento: §4 y §5

las respuestas las escribo yo, con mis palabras. si tengo que mirar el
notebook para contestar, significa que todavia no me la se.

## 1. ¿por que su accuracy es tan alta? ¿no hay sobreajuste?

donde esta la respuesta: §3.1 y §5.2. con 83% de trafico benigno la accuracy no sirve; la decision se toma con f1 macro

mi respuesta:

> (escribir aca)

## 2. ¿por que ese escalado en la etapa 1 y no otro?

donde esta la respuesta: §4.1. con transformacion por cuantiles el one-class svm daba roc-auc 0,38, peor que el azar, porque recorta justo lo que hay que detectar

mi respuesta:

> (escribir aca)

## 3. son 11 modelos: ¿que aporta cada uno? ¿no son de mas?

donde esta la respuesta: §4.2 y §4.3. la idea es que sean enfoques distintos, no 11 variantes de lo mismo. una frase por modelo

mi respuesta:

> (escribir aca)

## 4. ¿por que el mejor detector de la etapa 1 solo saca 22% de recall?

donde esta la respuesta: §5.1. el random forest supervisado saca 77%: esa brecha es lo que cuesta no tener ataques etiquetados

mi respuesta:

> (escribir aca)

## 5. ¿como escogieron el umbral del 1% de falsas alarmas?

donde esta la respuesta: §5.1. se fija sobre el conjunto de calibracion, nunca sobre el test. el 1% es lo que un analista tolera

mi respuesta:

> (escribir aca)

