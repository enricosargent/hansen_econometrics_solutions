"""Shared quantile-regression tools for Chapter 24 exercises.

The helpers build wage samples, fit conditional quantiles, and print compact
tables so scripts can focus on distributional comparisons.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg


PYTHON_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PYTHON_ROOT / "data"


def as_float(value: object) -> float:
    """Convert spreadsheet cells to floats while preserving missing values."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "NA", "."}:
            return float("nan")
        return float(text)
    return float(value)


def load_xlsx(name: str) -> list[dict[str, float | str]]:
    """Read a dataset workbook into row dictionaries for explicit sample filters."""
    path = DATA_ROOT / name / f"{name}.xlsx"
    frame = pd.read_excel(path)
    data: list[dict[str, float | str]] = []
    for raw_row in frame.to_dict(orient="records"):
        row: dict[str, float | str] = {}
        for key, value in raw_row.items():
            try:
                row[str(key)] = as_float(value)
            except (TypeError, ValueError):
                row[str(key)] = "" if pd.isna(value) else str(value)
        data.append(row)
    return data


def log_wage(row: dict[str, float | str]) -> float:
    """Compute log hourly wage, returning missing for nonpositive hours or wages."""
    earnings = float(row["earnings"])
    hours = float(row["hours"])
    weeks = float(row["week"])
    if hours <= 0 or weeks <= 0:
        return float("nan")
    wage = earnings / (hours * weeks)
    if wage <= 0 or not math.isfinite(wage):
        return float("nan")
    return math.log(wage)


def quantile_regression(
    y: np.ndarray,
    x: np.ndarray,
    tau: float,
    beta0: np.ndarray | None = None,
    *,
    final_smoothing: float = 1e-7,
    tolerance: float = 1e-6,
    max_iter: int = 20000,
) -> np.ndarray:
    """Fit one conditional quantile using statsmodels' linear-programming solver."""
    del beta0, final_smoothing
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    result = QuantReg(y, x).fit(q=tau, max_iter=max_iter, p_tol=tolerance)
    return np.asarray(result.params, dtype=float)


def fit_quantiles(
    y: np.ndarray,
    x: np.ndarray,
    taus: list[float],
    *,
    final_smoothing: float = 1e-7,
    tolerance: float = 1e-6,
    max_iter: int = 20000,
) -> np.ndarray:
    """Fit a sequence of quantiles, passing each estimate as the next warm start."""
    estimates = []
    beta0: np.ndarray | None = None
    for tau in taus:
        beta0 = quantile_regression(
            y,
            x,
            tau,
            beta0,
            final_smoothing=final_smoothing,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        estimates.append(beta0)
    return np.asarray(estimates)


def polynomial_design(experience: np.ndarray) -> np.ndarray:
    """Build Hansen's centered polynomial in experience for wage quantiles."""
    z = (np.asarray(experience, dtype=float) - 20.0) / 10.0
    return np.column_stack([np.ones(z.size), z, z**2, z**3, z**4, z**5])


def print_table(header: list[str], rows: list[list[float | int | str]]) -> None:
    """Print aligned text tables while leaving numerical calculations untouched."""
    widths = [len(column) for column in header]
    text_rows = []
    for row in rows:
        text_row = []
        for value in row:
            if isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = str(value)
            text_row.append(text)
        widths = [max(width, len(text)) for width, text in zip(widths, text_row)]
        text_rows.append(text_row)
    print("  ".join(column.ljust(width) for column, width in zip(header, widths)))
    for row in text_rows:
        print("  ".join(text.ljust(width) for text, width in zip(row, widths)))
