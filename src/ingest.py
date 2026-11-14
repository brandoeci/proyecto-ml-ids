"""
Descarga CSE-CIC-IDS2018 y arma la muestra de trabajo.

El dataset son 6.4 GB de CSV y no nos caben todos a la vez en el disco, asi que
por cada uno de los 10 archivos hacemos:
  1. bajarlo de S3 (el bucket es publico)
  2. leerlo por trozos y contar las etiquetas reales, antes de muestrear
  3. muestrear: nos quedamos con todos los ataques y con el 8% de los benignos
  4. guardarlo en parquet y borrar el CSV

Asi el disco no pasa de ~4.5 GB en ningun momento.

Los conteos completos quedan en data/processed/conteo_etiquetas_real.json, para
que en el EDA podamos mostrar el desbalance de verdad y no el de la muestra.

Uso:  python -I src/ingest.py              # los 10 dias
      python -I src/ingest.py <dia> ...    # solo esos dias, para reanudar
"""

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "cse-cic-ids2018"
PREFIX = "Processed Traffic Data for ML Algorithms"
SUFIJO = "_TrafficForML_CICFlowMeter.csv"

# El dataset viene separado por dia de captura y cada dia trae ataques distintos.
DIAS = [
    "Wednesday-14-02-2018",
    "Thursday-15-02-2018",
    "Friday-16-02-2018",
    "Thuesday-20-02-2018",   # 3.8 GB y con 4 columnas extra (asi viene de UNB)
    "Wednesday-21-02-2018",
    "Thursday-22-02-2018",
    "Friday-23-02-2018",
    "Wednesday-28-02-2018",
    "Thursday-01-03-2018",
    "Friday-02-03-2018",
]

# Estas 4 columnas solo aparecen en el archivo del 20-02. Las botamos porque son
# identificadores de red: si las dejamos, el modelo se puede aprender las IPs de
# los atacantes en vez del comportamiento del ataque.
COLS_EXTRA = ["Flow ID", "Src IP", "Src Port", "Dst IP"]

TASA_BENIGNO = 0.08      # nos quedamos con el 8% del trafico benigno
MAX_POR_CLASE = 100_000  # tope por ataque, para que no domine uno solo
TAM_TROZO = 400_000
SEMILLA = 42

RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_INTERIM = RAIZ / "data" / "interim"
DIR_PROC = RAIZ / "data" / "processed"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def descargar(dia):
    destino = DIR_RAW / f"{dia}{SUFIJO}"
    if destino.exists() and destino.stat().st_size > 0:
        log(f"  ya estaba descargado ({destino.stat().st_size / 1e6:.0f} MB)")
        return destino

    uri = f"s3://{BUCKET}/{PREFIX}/{dia}{SUFIJO}"
    log(f"  descargando {uri}")
    res = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", uri, str(destino)],
        capture_output=True,
        text=True,
    )
    # Si no revisamos esto, una descarga fallida pasa de largo y el dia queda vacio.
    if res.returncode != 0:
        raise RuntimeError(f"fallo la descarga de {dia}: {res.stderr.strip()}")
    log(f"  descargado ({destino.stat().st_size / 1e6:.0f} MB)")
    return destino


def limpiar_trozo(tr):
    """La limpieza minima, la que no admite discusion. El analisis va en el notebook."""
    # 1. Varios archivos traen la fila de encabezado repetida en la mitad.
    tr = tr[tr["Label"] != "Label"]

    # 2. Las columnas identificadoras del archivo del 20-02.
    sobrantes = [c for c in COLS_EXTRA if c in tr.columns]
    if sobrantes:
        tr = tr.drop(columns=sobrantes)

    # 3. Todo lo que no sea Timestamp ni Label tiene que ser numerico. Hay celdas
    #    con "Infinity" y con texto suelto, esas quedan en NaN.
    no_num = ["Timestamp", "Label"]
    cols_num = [c for c in tr.columns if c not in no_num]
    for c in cols_num:
        if tr[c].dtype == object:
            tr[c] = pd.to_numeric(tr[c], errors="coerce")
    tr[cols_num] = tr[cols_num].replace([np.inf, -np.inf], np.nan)
    # float32 en vez de float64: con esto el dataset ocupa la mitad de RAM.
    tr[cols_num] = tr[cols_num].astype("float32")

    return tr


def procesar(dia, csv, rng):
    conteo = Counter()
    piezas = []
    filas_leidas = 0

    lector = pd.read_csv(csv, chunksize=TAM_TROZO, low_memory=False)
    for i, tr in enumerate(lector):
        filas_leidas += len(tr)
        tr = limpiar_trozo(tr)

        # Contamos antes de muestrear, si no perdemos el desbalance real.
        conteo.update(tr["Label"].value_counts().to_dict())

        es_benigno = tr["Label"] == "Benign"
        ataques = tr[~es_benigno]
        benignos = tr[es_benigno]
        if len(benignos):
            mascara = rng.random(len(benignos)) < TASA_BENIGNO
            benignos = benignos[mascara]

        piezas.append(pd.concat([ataques, benignos], ignore_index=True))
        if (i + 1) % 5 == 0:
            log(f"    {filas_leidas:,} filas leidas")

    df = pd.concat(piezas, ignore_index=True)

    # Tope por ataque. Se aplica dentro del dia, que alcanza porque cada ataque
    # ocurre solo en uno o dos dias.
    partes = []
    for etiqueta, grupo in df.groupby("Label", observed=True):
        if etiqueta != "Benign" and len(grupo) > MAX_POR_CLASE:
            grupo = grupo.sample(MAX_POR_CLASE, random_state=SEMILLA)
        partes.append(grupo)
    df = pd.concat(partes, ignore_index=True)

    df["dia_captura"] = dia
    return df, conteo


def main():
    for d in (DIR_RAW, DIR_INTERIM, DIR_PROC):
        d.mkdir(parents=True, exist_ok=True)

    pedidos = sys.argv[1:]
    if pedidos:
        desconocidos = [d for d in pedidos if d not in DIAS]
        if desconocidos:
            log(f"dias no reconocidos: {desconocidos}")
            log(f"validos: {DIAS}")
            return
        dias = pedidos
    else:
        dias = DIAS

    rng = np.random.default_rng(SEMILLA)
    conteo_total = Counter()
    conteo_por_dia = {}
    filas_muestra = 0

    # Si ya corrimos antes, arrastramos los conteos para no perderlos. Hay que
    # leer el archivo UNA sola vez aca arriba: lo reescribimos despues de cada
    # dia, asi que si lo consultamos mas adelante ya no tiene lo original.
    # (Esto nos costo perder los conteos de un dia la primera vez.)
    archivo_conteo = DIR_PROC / "conteo_etiquetas_real.json"
    conteo_previo = {}
    if archivo_conteo.exists():
        guardado = json.loads(archivo_conteo.read_text(encoding="utf-8"))
        conteo_previo = guardado.get("por_dia", {})
        conteo_por_dia = {k: v for k, v in conteo_previo.items() if k not in dias}
        for v in conteo_por_dia.values():
            conteo_total.update(v)

    for n, dia in enumerate(dias, 1):
        log(f"=== [{n}/{len(dias)}] {dia} ===")
        salida = DIR_INTERIM / f"{dia}.parquet"

        if salida.exists():
            if dia in conteo_previo:
                log("  el parquet ya existe, lo saltamos (borralo para reprocesar)")
                conteo_por_dia[dia] = conteo_previo[dia]
                conteo_total.update(conteo_previo[dia])
                filas_muestra += len(pd.read_parquet(salida, columns=["Label"]))
                continue
            # Esta el parquet pero no los conteos de ese dia, hay que reprocesar
            # o la distribucion real queda incompleta.
            log("  el parquet existe pero sin conteos, se reprocesa")

        csv = descargar(dia)
        try:
            df, conteo = procesar(dia, csv, rng)
        finally:
            # El CSV se borra siempre, pase lo que pase. Es lo que salva el disco.
            if csv.exists():
                csv.unlink()
                log("  CSV borrado")

        df.to_parquet(salida, index=False)
        filas_muestra += len(df)
        conteo_total.update(conteo)
        conteo_por_dia[dia] = dict(conteo)

        log(f"  muestra: {len(df):,} filas -> {salida.name} "
            f"({salida.stat().st_size / 1e6:.0f} MB)")
        log(f"  etiquetas del dia: {dict(conteo)}")

        # Guardamos despues de cada dia por si se corta la corrida.
        archivo_conteo.write_text(
            json.dumps(
                {"total": dict(conteo_total), "por_dia": conteo_por_dia},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    log("=== resumen ===")
    log(f"filas reales en el dataset completo: {sum(conteo_total.values()):,}")
    log(f"filas en la muestra de trabajo:      {filas_muestra:,}")
    for etiqueta, k in sorted(conteo_total.items(), key=lambda x: -x[1]):
        log(f"  {etiqueta:<28} {k:>12,}")


if __name__ == "__main__":
    main()
