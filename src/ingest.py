"""
Ingesta de CSE-CIC-IDS2018 con disco limitado.

Estrategia: por cada archivo del dataset (10 en total, 6.4 GB)
  1. descargar el CSV desde S3 (acceso anonimo)
  2. leerlo por trozos y contar la distribucion REAL de etiquetas (sin muestrear)
  3. muestrear: conservar todos los flujos de ataque, submuestrear los benignos
  4. guardar en Parquet y BORRAR el CSV crudo

Asi el pico de disco se queda por debajo de ~4.5 GB en vez de los 6.4 GB
que ocuparia bajar todo de una, y el resultado final cabe en memoria.

Las cuentas completas se guardan en data/processed/conteo_etiquetas_real.json
para que el EDA pueda mostrar el desbalance verdadero aunque el modelado
se haga sobre la muestra.

Uso:  python -I src/ingest.py              # los 10 dias
      python -I src/ingest.py <dia> ...    # solo esos dias (util para reanudar)
"""

from __future__ import annotations

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

# El dataset esta organizado por dia de captura. Cada dia trae ataques distintos.
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

# Columnas identificadoras que solo aparecen en el archivo del 20-02.
# Se descartan: son identificadores de red, no comportamiento. Si se dejaran,
# el modelo podria memorizar IPs atacantes en vez de aprender el ataque.
COLS_EXTRA = ["Flow ID", "Src IP", "Src Port", "Dst IP"]

TASA_BENIGNO = 0.08      # conservamos 8% del trafico benigno
MAX_POR_CLASE = 100_000  # tope por clase de ataque, para que no domine una sola
TAM_TROZO = 400_000
SEMILLA = 42

RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_INTERIM = RAIZ / "data" / "interim"
DIR_PROC = RAIZ / "data" / "processed"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def descargar(dia: str) -> Path:
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
    if res.returncode != 0:
        raise RuntimeError(f"fallo la descarga de {dia}: {res.stderr.strip()}")
    log(f"  descargado ({destino.stat().st_size / 1e6:.0f} MB)")
    return destino


def limpiar_trozo(tr: pd.DataFrame) -> pd.DataFrame:
    """Limpieza minima e indiscutible, a nivel de trozo."""
    # 1. Algunos archivos traen la fila de encabezado repetida en medio.
    tr = tr[tr["Label"] != "Label"]

    # 2. Columnas identificadoras del archivo del 20-02.
    sobrantes = [c for c in COLS_EXTRA if c in tr.columns]
    if sobrantes:
        tr = tr.drop(columns=sobrantes)

    # 3. Todo lo que no sea Timestamp ni Label debe ser numerico.
    #    Hay celdas con "Infinity" y con texto suelto -> se vuelven NaN.
    no_num = ["Timestamp", "Label"]
    cols_num = [c for c in tr.columns if c not in no_num]
    for c in cols_num:
        if tr[c].dtype == object:
            tr[c] = pd.to_numeric(tr[c], errors="coerce")
    tr[cols_num] = tr[cols_num].replace([np.inf, -np.inf], np.nan)
    tr[cols_num] = tr[cols_num].astype("float32")

    return tr


def procesar(dia: str, csv: Path, rng: np.random.Generator) -> tuple[pd.DataFrame, Counter]:
    conteo = Counter()
    piezas: list[pd.DataFrame] = []
    filas_leidas = 0

    lector = pd.read_csv(csv, chunksize=TAM_TROZO, low_memory=False)
    for i, tr in enumerate(lector):
        filas_leidas += len(tr)
        tr = limpiar_trozo(tr)

        # Distribucion REAL, antes de muestrear.
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

    # Tope por clase de ataque (global dentro del dia; cada ataque vive en 1-2 dias).
    partes = []
    for etiqueta, grupo in df.groupby("Label", observed=True):
        if etiqueta != "Benign" and len(grupo) > MAX_POR_CLASE:
            grupo = grupo.sample(MAX_POR_CLASE, random_state=SEMILLA)
        partes.append(grupo)
    df = pd.concat(partes, ignore_index=True)

    df["dia_captura"] = dia
    return df, conteo


def main() -> int:
    for d in (DIR_RAW, DIR_INTERIM, DIR_PROC):
        d.mkdir(parents=True, exist_ok=True)

    pedidos = sys.argv[1:]
    if pedidos:
        desconocidos = [d for d in pedidos if d not in DIAS]
        if desconocidos:
            log(f"dias no reconocidos: {desconocidos}")
            log(f"validos: {DIAS}")
            return 2
        dias = pedidos
    else:
        dias = DIAS

    rng = np.random.default_rng(SEMILLA)
    conteo_total: Counter = Counter()
    conteo_por_dia: dict[str, dict] = {}
    filas_muestra = 0

    # Si ya hay conteos de una corrida previa, se arrastran para no perderlos.
    # Se lee UNA sola vez: el archivo se reescribe tras cada dia, asi que
    # consultarlo mas adelante ya no devolveria el contenido original.
    archivo_conteo = DIR_PROC / "conteo_etiquetas_real.json"
    conteo_previo: dict[str, dict] = {}
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
                log("  parquet ya existe, se omite (borralo para reprocesar)")
                conteo_por_dia[dia] = conteo_previo[dia]
                conteo_total.update(conteo_previo[dia])
                filas_muestra += len(pd.read_parquet(salida, columns=["Label"]))
                continue
            # El parquet existe pero no hay conteos guardados de ese dia: se
            # reprocesa, porque si no la distribucion real quedaria incompleta.
            log("  parquet existe pero sin conteos guardados -> se reprocesa")

        csv = descargar(dia)
        try:
            df, conteo = procesar(dia, csv, rng)
        finally:
            # El CSV crudo se borra siempre: es lo que mantiene el disco a salvo.
            if csv.exists():
                csv.unlink()
                log("  CSV crudo borrado")

        df.to_parquet(salida, index=False, compression="zstd")
        filas_muestra += len(df)
        conteo_total.update(conteo)
        conteo_por_dia[dia] = dict(conteo)

        log(f"  muestra: {len(df):,} filas -> {salida.name} "
            f"({salida.stat().st_size / 1e6:.0f} MB)")
        log(f"  etiquetas del dia: {dict(conteo)}")

        # Se va guardando por si se corta la corrida.
        (DIR_PROC / "conteo_etiquetas_real.json").write_text(
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
