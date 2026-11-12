"""Parametros compartidos por el notebook. Un solo sitio donde tocar."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROC = RAIZ / "data" / "processed"
DIR_FIG = RAIZ / "reports" / "figuras"
RUTA_DATOS = DIR_PROC / "ids2018_modelado.parquet"

SEMILLA = 42

# Columnas que no son variables predictoras.
NO_FEATURES = ["Timestamp", "Label", "familia", "es_ataque", "dia_captura", "particion"]

# Variables con fuga de informacion: identifican el ataque por donde ocurre,
# no por como se comporta. Se excluyen del modelo final (ver seccion de EDA).
COLS_FUGA = ["Dst Port"]

# Topes de tamano por modelo. Los modelos basados en distancias o kernels no
# escalan a millones de filas en un portatil, asi que se entrenan sobre
# submuestras aleatorias estratificadas. Queda declarado para que sea auditable.
N_BENIGNO_E1 = 150_000   # Isolation Forest
N_OCSVM = 20_000         # One-Class SVM  (coste ~O(n^2))
N_LOF = 50_000           # Local Outlier Factor
N_AUTOENCODER = 100_000  # autoencoder (MLPRegressor)
N_E2_TRAIN = 120_000     # etapa 2: arboles y boosting
N_E2_LENTOS = 40_000     # etapa 2: kNN y MLP
N_TEST_BENIGNO = 150_000 # tope de benignos en test (los ataques se usan todos)
N_BUSQUEDA = 60_000      # submuestra para la busqueda de hiperparametros

# Punto de operacion de la etapa 1: presupuesto de falsas alarmas.
FPR_OBJETIVO = 0.01      # 1% de los flujos benignos puede generar alerta

# Etapa 2: cobertura que se exige sobre familias conocidas al fijar el umbral
# de rechazo. 0.95 => se acepta perder 5% de los aciertos para poder decir
# "no se" ante algo desconocido.
COBERTURA_CONOCIDOS = 0.95

FAMILIAS_ATAQUE = ["Fuerza bruta", "DoS", "DDoS", "Ataque web", "Infiltracion", "Botnet"]
ETIQUETA_DESCONOCIDO = "Desconocido"

N_E2_TEST = 60_000       # tope del test de la etapa 2 (kNN no escala mas alla)
