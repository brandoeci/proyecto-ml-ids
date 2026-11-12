"""
Consolidacion de los parquets por dia en un unico dataset de modelado.

Hace solo lo mecanico e indiscutible. El analisis de variables (constantes,
correlacionadas, fuga por Dst Port) se queda en el notebook, porque es
analisis y debe verse.

Pasos:
  1. unir los 10 parquets por dia
  2. normalizar las etiquetas y agruparlas en familias de ataque
  3. parsear Timestamp (viene en dd/mm/yyyy, con formatos mezclados)
  4. quitar duplicados exactos
  5. asignar particion train/test por tiempo DENTRO de cada dia
  6. guardar data/processed/ids2018_modelado.parquet + metadatos

Uso:  python -I src/preparar.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DIR_INTERIM = RAIZ / "data" / "interim"
DIR_PROC = RAIZ / "data" / "processed"

FRAC_TRAIN = 0.70   # primer 70% de cada dia -> train, ultimo 30% -> test
SEMILLA = 42

# --- Taxonomia -------------------------------------------------------------
# Las 14 etiquetas de ataque de CSE-CIC-IDS2018 se agrupan en 6 familias.
# Motivo: varias etiquetas son solo la herramienta usada (LOIC, HOIC, GoldenEye,
# Hulk) para el mismo comportamiento. Para un analista de un SOC lo que importa
# es la familia; y agrupar le da a la etapa 2 clases con suficientes ejemplos.
#
# Las llaves estan en minuscula y sin espacios para que el emparejamiento
# aguante las variantes de escritura del dataset (que no es consistente, e
# incluye el error tipografico "Infilteration").
FAMILIAS = {
    "benign": "Benigno",

    "ftp-bruteforce": "Fuerza bruta",
    "ssh-bruteforce": "Fuerza bruta",

    "dosattacks-goldeneye": "DoS",
    "dosattacks-slowloris": "DoS",
    "dosattacks-slowhttptest": "DoS",
    "dosattacks-hulk": "DoS",

    "ddosattack-loic-udp": "DDoS",
    "ddosattacks-loic-http": "DDoS",
    "ddosattack-hoic": "DDoS",

    "bruteforce-web": "Ataque web",
    "bruteforce-xss": "Ataque web",
    "sqlinjection": "Ataque web",

    "infilteration": "Infiltracion",
    "infiltration": "Infiltracion",

    "bot": "Botnet",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalizar(etiqueta: str) -> str:
    """Pasa una etiqueta cruda a la llave canonica de FAMILIAS."""
    return "".join(str(etiqueta).lower().split())


def a_familia(etiqueta: str) -> str:
    llave = normalizar(etiqueta)
    if llave not in FAMILIAS:
        raise KeyError(
            f"etiqueta sin familia asignada: {etiqueta!r} (normalizada: {llave!r}). "
            "Agregala al diccionario FAMILIAS antes de seguir."
        )
    return FAMILIAS[llave]


def main() -> int:
    DIR_PROC.mkdir(parents=True, exist_ok=True)

    archivos = sorted(DIR_INTERIM.glob("*.parquet"))
    if not archivos:
        log("no hay parquets en data/interim. Corre primero src/ingest.py")
        return 1
    log(f"uniendo {len(archivos)} parquets por dia")

    df = pd.concat([pd.read_parquet(a) for a in archivos], ignore_index=True)
    log(f"filas unidas: {len(df):,} | columnas: {df.shape[1]}")

    # --- etiquetas ---------------------------------------------------------
    crudas = sorted(df["Label"].unique())
    log(f"etiquetas crudas encontradas ({len(crudas)}): {crudas}")

    df["familia"] = df["Label"].map(a_familia)
    df["es_ataque"] = (df["familia"] != "Benigno").astype("int8")

    log("filas por familia:")
    for fam, k in df["familia"].value_counts().items():
        log(f"  {fam:<16} {k:>10,}")

    # --- tiempo ------------------------------------------------------------
    # El dataset mezcla formatos; dayfirst=True porque viene en dd/mm/yyyy.
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], dayfirst=True, format="mixed", errors="coerce"
    )
    sin_fecha = int(df["Timestamp"].isna().sum())
    if sin_fecha:
        log(f"ADVERTENCIA: {sin_fecha:,} filas sin Timestamp parseable -> se descartan")
        df = df[df["Timestamp"].notna()].copy()

    # --- duplicados exactos ------------------------------------------------
    # CSE-CIC-IDS2018 trae muchisimas filas repetidas (rafagas de ataque donde
    # los flujos salen identicos). Si no se quitan, el mismo flujo cae en train
    # y en test y las metricas salen infladas.
    cols_feat = [c for c in df.columns
                 if c not in ("Timestamp", "Label", "familia", "es_ataque", "dia_captura")]
    antes = len(df)
    conteo_antes_dedup = df["Label"].value_counts().to_dict()
    df = df.drop_duplicates(subset=cols_feat + ["Label"], keep="first")
    conteo_tras_dedup = df["Label"].value_counts().to_dict()
    log(f"duplicados exactos eliminados: {antes - len(df):,} "
        f"({(antes - len(df)) / antes:.1%} de la muestra) -> quedan {len(df):,}")

    # El agregado esconde lo importante: la tasa de duplicados varia muchisimo
    # entre ataques. Hay etiquetas que practicamente desaparecen.
    log("tasa de duplicados por etiqueta:")
    tasas = sorted(
        ((e, conteo_antes_dedup[e], conteo_tras_dedup.get(e, 0)) for e in conteo_antes_dedup),
        key=lambda x: -(1 - x[2] / x[1]),
    )
    for etiqueta, n_antes, n_desp in tasas:
        pct = (1 - n_desp / n_antes) * 100
        aviso = "  <-- casi no queda nada" if pct > 99 else ""
        log(f"  {etiqueta:<26} {n_antes:>9,} -> {n_desp:>8,}  ({pct:5.1f}% duplicados){aviso}")

    # --- particion temporal estratificada ----------------------------------
    # No se usa un split aleatorio: los flujos de una misma rafaga de ataque son
    # casi identicos entre si, y repartidos al azar el modelo ve en test algo
    # practicamente igual a lo que vio en train. Partir por tiempo evita eso.
    #
    # Se parte por tiempo DENTRO de cada par (dia, familia), no por dia a secas:
    #   - un corte global por dias dejaria familias completas fuera del train,
    #     porque cada ataque ocurre en dias especificos
    #   - un corte por dia a secas vacia las familias cuyos flujos unicos estan
    #     concentrados al inicio de su dia (le pasaba a Fuerza bruta: sus flujos
    #     son tan repetitivos que el deduplicado deja solo los primeros, y en
    #     test quedaban 53 de 50.252)
    # Partiendo dentro de cada (dia, familia) se conserva lo importante -el test
    # ocurre despues del train para esa misma familia, asi que ninguna rafaga se
    # reparte entre ambos- y cada familia queda con ~30% para evaluar.
    df = df.sort_values(["dia_captura", "familia", "Timestamp"],
                        kind="stable").reset_index(drop=True)
    df["particion"] = "test"
    for _, idx in df.groupby(["dia_captura", "familia"], observed=True).groups.items():
        idx = list(idx)
        corte = int(len(idx) * FRAC_TRAIN)
        df.loc[idx[:corte], "particion"] = "train"

    log("reparto train/test:")
    tabla = pd.crosstab(df["familia"], df["particion"])
    for fam in tabla.index:
        tr = int(tabla.loc[fam].get("train", 0))
        te = int(tabla.loc[fam].get("test", 0))
        log(f"  {fam:<16} train={tr:>9,}  test={te:>9,}")

    # Una familia sin ejemplos en alguna particion romperia el modelado.
    for fam in tabla.index:
        if int(tabla.loc[fam].get("train", 0)) == 0 or int(tabla.loc[fam].get("test", 0)) == 0:
            log(f"ADVERTENCIA: la familia {fam!r} quedo vacia en una particion")

    # --- guardar -----------------------------------------------------------
    salida = DIR_PROC / "ids2018_modelado.parquet"
    df.to_parquet(salida, index=False, compression="zstd")
    log(f"guardado {salida} ({salida.stat().st_size / 1e6:.0f} MB)")

    meta = {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "etiquetas_crudas": crudas,
        "familias": {k: int(v) for k, v in df["familia"].value_counts().items()},
        "duplicados_eliminados": int(antes - len(df)),
        "conteo_antes_dedup": {k: int(v) for k, v in conteo_antes_dedup.items()},
        "conteo_tras_dedup": {k: int(v) for k, v in conteo_tras_dedup.items()},
        "frac_train": FRAC_TRAIN,
        "rango_fechas": [str(df["Timestamp"].min()), str(df["Timestamp"].max())],
        "reparto": {f: {p: int(tabla.loc[f].get(p, 0)) for p in tabla.columns}
                    for f in tabla.index},
    }
    (DIR_PROC / "metadatos_modelado.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("listo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
