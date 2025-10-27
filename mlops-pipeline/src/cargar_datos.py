# cargar_datos.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _project_root(start: Path | None = None) -> Path:
    """Intenta localizar la raíz del proyecto buscando pyproject.toml o .git."""
    base = (start or Path(__file__).resolve()).parent
    p = base
    for _ in range(10):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd()

def _resolve_input(path_like: str | Path) -> Path:
    """Resuelve el path del Excel de forma robusta (absoluta)."""
    p = Path(path_like)
    if p.is_absolute() and p.exists():
        return p
    # 1) relativo a la raíz del proyecto
    root = _project_root()
    cand = (root / p)
    if cand.exists():
        return cand
    # 2) relativo al archivo actual
    here = Path(__file__).resolve().parent
    cand2 = (here / p)
    if cand2.exists():
        return cand2
    # 3) relativo al CWD (último intento)
    cand3 = p.resolve()
    if cand3.exists():
        return cand3
    raise FileNotFoundError(
        f"No se encontró el archivo Excel.\n"
        f"Probed:\n - {cand}\n - {cand2}\n - {cand3}"
    )

def cargar_dataset(
    raw_path: str | Path = "BD_creditos.xlsx",
    cache_path: str | Path = "creditos.pkl",
    usar_cache: bool = False,
    sheet_name: int | str = 0,
) -> pd.DataFrame:
    """
    Si usar_cache=True y existe el pickle, lo carga.
    Si no, lee el Excel y (opcionalmente) guarda el pickle.
    """
    cache_path = Path(cache_path)
    if usar_cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    xlsx_path = _resolve_input(raw_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

    # Guardar caché de forma segura (crear carpetas si no existen)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)

    return df

