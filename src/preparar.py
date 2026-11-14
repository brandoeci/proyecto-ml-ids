"""
Junta los parquets de cada dia en un solo dataset para modelar.

Aca solo va lo mecanico. El analisis de variables (constantes, correlacionadas,
la fuga por Dst Port) se queda en el notebook, porque eso es analisis y hay que
poder verlo.

Lo que hace:
  1. une los 10 parquets por dia
  2. normaliza las etiquetas y las agrupa en familias
  3. parsea Timestamp (viene en dd/mm/yyyy y con formatos mezclados)
  4. quita duplicados exactos
  5. parte en train/test por tiempo
  6. guarda data/processed/ids2018_modelado.parquet y los metadatos

Uso:  python -I src/preparar.py
"""

import json
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DIR_INTERIM = RAIZ / "data" / "interim"
DIR_PROC = RAIZ / "data" / "processed"

FRAC_TRAIN = 0.70   # el primer 70% de cada dia va a train, el ultimo 30% a test
SEMILLA = 42

# Agrupamos las 14 etiquetas de ataque en 6 familias. La razon es que varias
# etiquetas son solo la herramienta que usaron (LOIC, HOIC, GoldenEye, Hulk) para
# hacer lo mismo. A un analista lo que le importa es la familia, y de paso le deja
# a la etapa 2 clases con suficientes ejemplos.
#
# Las llaves van en minuscula y sin espacios para que el emparejamiento aguante
# las variantes de escritura del dataset, que no es consistente e incluso trae la
# etiqueta mal escrita ("Infilteration").
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def normalizar(etiqueta):
    """Deja la etiqueta como la llave que usamos en FAMILIAS."""
    return "".join(str(etiqueta).lower().split())


def a_familia(etiqueta):
    llave = normalizar(etiqueta)
    if llave not in FAMILIAS:
        raise KeyError(
            f"etiqueta sin familia asignada: {etiqueta!r} (normalizada: {llave!r}). "
            "Hay que agregarla al diccionario FAMILIAS antes de seguir."
        )
    return FAMILIAS[llave]


def main():
    DIR_PROC.mkdir(parents=True, exist_ok=True)

    archivos = sorted(DIR_INTERIM.glob("*.parquet"))
    if not archivos:
        log("no hay parquets en data/interim. Corre primero src/ingest.py")
        return
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
    # El dataset mezcla formatos; dayfirst porque viene en dd/mm/yyyy.
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], dayfirst=True, format="mixed", errors="coerce"
    )
    sin_fecha = int(df["Timestamp"].isna().sum())
    if sin_fecha:
        log(f"OJO: {sin_fecha:,} filas sin Timestamp parseable, se descartan")
        df = df[df["Timestamp"].notna()].copy()

    # --- duplicados exactos ------------------------------------------------
    # El dataset trae muchisimas filas repetidas, sobre todo en las rafagas de
    # ataque, donde los flujos salen identicos. Si no las quitamos, el mismo
    # flujo cae en train y en test y las metricas salen infladas.
    cols_feat = [c for c in df.columns
                 if c not in ("Timestamp", "Label", "familia", "es_ataque", "dia_captura")]
    antes = len(df)
    conteo_antes_dedup = df["Label"].value_counts().to_dict()
    df = df.drop_duplicates(subset=cols_feat + ["Label"], keep="first")
    conteo_tras_dedup = df["Label"].value_counts().to_dict()
    log(f"duplicados exactos eliminados: {antes - len(df):,} "
        f"({(antes - len(df)) / antes:.1%} de la muestra) -> quedan {len(df):,}")

    # El promedio esconde lo importante: la tasa cambia muchisimo de un ataque a
    # otro. Hay etiquetas que practicamente desaparecen.
    log("tasa de duplicados por etiqueta:")
    tasas = sorted(
        ((e, conteo_antes_dedup[e], conteo_tras_dedup.get(e, 0)) for e in conteo_antes_dedup),
        key=lambda x: -(1 - x[2] / x[1]),
    )
    for etiqueta, n_antes, n_desp in tasas:
        pct = (1 - n_desp / n_antes) * 100
        aviso = "  <-- casi no queda nada" if pct > 99 else ""
        log(f"  {etiqueta:<26} {n_antes:>9,} -> {n_desp:>8,}  ({pct:5.1f}% duplicados){aviso}")

    # --- particion train/test por tiempo -----------------------------------
    # No usamos split aleatorio: los flujos de una misma rafaga de ataque son casi
    # identicos entre si, y repartidos al azar el modelo ve en test algo
    # practicamente igual a lo que ya vio entrenando.
    #
    # Partimos por tiempo dentro de cada par (dia, familia), no por dia a secas.
    # Probamos las dos otras formas y ninguna sirve:
    #   - cortar por dias completos deja familias enteras fuera del train, porque
    #     cada ataque ocurre en dias especificos
    #   - cortar por dia a secas vacia las familias cuyos flujos unicos quedaron
    #     al comienzo del dia. A Fuerza bruta le pasaba: es tan repetitiva que el
    #     deduplicado deja solo los primeros, y en test quedaban 53 de 50.252
    # Dentro de cada (dia, familia) se conserva lo que importa, que el test ocurra
    # despues del train para esa misma familia, y cada familia queda con ~30% para
    # evaluar.
    #
    # El sort va con kind="stable": si no, las filas con el mismo Timestamp pueden
    # quedar en otro orden entre corridas y el split deja de ser reproducible.
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

    # Si una familia queda vacia en alguna particion, el modelado se rompe.
    for fam in tabla.index:
        if int(tabla.loc[fam].get("train", 0)) == 0 or int(tabla.loc[fam].get("test", 0)) == 0:
            log(f"OJO: la familia {fam!r} quedo vacia en una particion")

    # --- guardar -----------------------------------------------------------
    salida = DIR_PROC / "ids2018_modelado.parquet"
    df.to_parquet(salida, index=False)
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


if __name__ == "__main__":
    main()
