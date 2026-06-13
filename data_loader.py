"""Dataset loader shared by the empirical exercise scripts.

The loader prefers Stata files when available and falls back to Excel or text,
letting each exercise request a dataset by name rather than hard-coded paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PYTHON_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PYTHON_ROOT / "data"


def load_dataset(name: str) -> pd.DataFrame:
    """Load a named dataset from the local empirical-data mirror.

    Stata files preserve Hansen's numeric codings best, so they are tried first;
    Excel and tab-delimited text are fallbacks for datasets without `.dta` files.
    """
    dataset_dir = DATA_ROOT / name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dta_files = sorted(dataset_dir.glob("*.dta"))
    if dta_files:
        try:
            return pd.read_stata(dta_files[0], convert_categoricals=False)
        except (ImportError, ValueError):
            pass

    xlsx_files = sorted(dataset_dir.glob("*.xlsx"))
    if xlsx_files:
        return pd.read_excel(xlsx_files[0])

    txt_files = sorted(dataset_dir.glob("*.txt"))
    if txt_files:
        return pd.read_csv(txt_files[0], sep="\t")

    raise FileNotFoundError(f"No supported data file found in {dataset_dir}")
