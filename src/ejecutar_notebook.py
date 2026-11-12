"""
Ejecuta el notebook de punta a punta y lo guarda con las salidas dentro.

Se usa nbclient directamente en vez del comando `jupyter nbconvert` porque este
ultimo no siempre queda en el PATH, y entonces el fallo pasa desapercibido.
Aqui, si una celda revienta, el proceso termina con codigo distinto de cero y
dice en que celda fue.

Uso:  python src/ejecutar_notebook.py [ruta_al_ipynb]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

RAIZ = Path(__file__).resolve().parent.parent
POR_DEFECTO = RAIZ / "notebooks" / "proyecto_ids.ipynb"
TIMEOUT_CELDA = 3600  # una hora por celda, de sobra para las busquedas


def main() -> int:
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else POR_DEFECTO
    if not ruta.exists():
        print(f"no existe: {ruta}")
        return 1

    nb = nbformat.read(ruta, as_version=4)
    n_cod = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"ejecutando {ruta.name}: {len(nb.cells)} celdas ({n_cod} de codigo)")
    print(f"directorio de trabajo del kernel: {ruta.parent}")

    cliente = NotebookClient(
        nb,
        timeout=TIMEOUT_CELDA,
        kernel_name="python3",
        resources={"metadata": {"path": str(ruta.parent)}},
        allow_errors=False,
    )

    t0 = time.perf_counter()
    try:
        cliente.execute()
    except CellExecutionError as e:
        nbformat.write(nb, ruta)  # se guarda igual, para poder ver donde fallo
        print(f"\nFALLO tras {time.perf_counter() - t0:.0f}s")
        print(str(e)[:4000])
        return 1

    nbformat.write(nb, ruta)
    transcurrido = time.perf_counter() - t0

    ejecutadas = sum(1 for c in nb.cells
                     if c.cell_type == "code" and c.get("execution_count"))
    con_salida = sum(1 for c in nb.cells
                     if c.cell_type == "code" and c.get("outputs"))
    print(f"\nlisto en {transcurrido / 60:.1f} min")
    print(f"  celdas ejecutadas : {ejecutadas}/{n_cod}")
    print(f"  celdas con salida : {con_salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
