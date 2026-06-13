"""Shared VAR and structural-IRF tools for Chapter 15 exercises.

The helpers centralize data loading, lag selection, moving-average recursion,
and identification calculations so each exercise can emphasize interpretation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


PYTHON_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PYTHON_ROOT / "data"


def load_excel(*parts: str) -> pd.DataFrame:
    """Load a workbook and parse a `time` column when the dataset provides one."""
    df = pd.read_excel(DATA_DIR.joinpath(*parts))
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df.copy()


def to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric values, turning nonnumeric entries into NA."""
    return pd.to_numeric(series, errors="coerce")


def log_level(series: pd.Series, scale: float = 100.0) -> pd.Series:
    """Express a level variable on the common 100 times log scale."""
    return scale * np.log(to_numeric(series))


def log_growth(series: pd.Series, order: int = 1, scale: float = 100.0) -> pd.Series:
    """Compute log growth rates in percentage-point units."""
    return log_level(series, scale=scale).diff(order)


def fit_var(
    data: pd.DataFrame,
    lags: int,
    *,
    trend: str = "c",
    exog: pd.DataFrame | None = None,
):
    """Fit the reduced-form VAR used before any structural identification."""
    model = VAR(data, exog=exog)
    return model.fit(lags, trend=trend)


def aic_table(
    data: pd.DataFrame,
    maxlags: int,
    *,
    trend: str = "c",
    exog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare VAR lag lengths using the Gaussian AIC reported by statsmodels."""
    rows = []
    model = VAR(data, exog=exog)
    for lag in range(1, maxlags + 1):
        result = model.fit(lag, trend=trend)
        rows.append({"lag": lag, "aic": result.aic})
    return pd.DataFrame(rows)


def ma_matrices(coefs: np.ndarray, horizons: int) -> np.ndarray:
    """Convert VAR coefficients into moving-average response matrices."""
    p, m, _ = coefs.shape
    theta = np.zeros((horizons + 1, m, m))
    theta[0] = np.eye(m)
    for h in range(1, horizons + 1):
        acc = np.zeros((m, m))
        for j in range(1, min(p, h) + 1):
            acc += coefs[j - 1] @ theta[h - j]
        theta[h] = acc
    return theta


def orthogonalized_irf(result, horizons: int):
    """Identify reduced-form shocks by the Cholesky factor of their covariance."""
    theta = ma_matrices(result.coefs, horizons)
    sigma = np.asarray(result.sigma_u, dtype=float)
    b = np.linalg.cholesky(sigma)
    return theta, theta @ b, b


def short_run_structural_irf(result, a_matrix: np.ndarray, horizons: int):
    """Compute structural impulse responses from short-run restrictions on A."""
    theta = ma_matrices(result.coefs, horizons)
    sigma = np.asarray(result.sigma_u, dtype=float)
    d = a_matrix @ sigma @ a_matrix.T
    b = np.linalg.solve(a_matrix, np.diag(np.sqrt(np.diag(d))))
    return theta, theta @ b, b, d


def long_run_structural_irf(result, horizons: int):
    """Identify permanent shocks using the long-run multiplier matrix."""
    theta = ma_matrices(result.coefs, horizons)
    sigma = np.asarray(result.sigma_u, dtype=float)
    a1 = np.eye(result.neqs) - result.coefs.sum(axis=0)
    c = np.linalg.cholesky(np.linalg.inv(a1) @ sigma @ np.linalg.inv(a1).T)
    b = a1 @ c
    return theta, theta @ b, a1, c, b


def solve_short_run_a_1517(sigma: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Solve the Exercise 15.17 contemporaneous matrix from covariance moments."""
    sigma = np.asarray(sigma, dtype=float)
    r1 = np.array([1.0, 0.0, -1.0, 0.0])
    v1 = r1 @ sigma

    a23 = -v1[1] / v1[2]
    r2 = np.array([0.0, 1.0, a23, 0.0])
    v2 = r2 @ sigma

    system_23 = np.array([[v1[0], v1[1]], [v2[0], v2[1]]], dtype=float)
    rhs_23 = -np.array([v1[2], v2[2]], dtype=float)
    a31, a32 = np.linalg.solve(system_23, rhs_23)
    r3 = np.array([a31, a32, 1.0, 0.0])
    v3 = r3 @ sigma

    system_4 = np.array(
        [
            [v1[0], v1[1], v1[2]],
            [v2[0], v2[1], v2[2]],
            [v3[0], v3[1], v3[2]],
        ],
        dtype=float,
    )
    rhs_4 = -np.array([v1[3], v2[3], v3[3]], dtype=float)
    a41, a42, a43 = np.linalg.solve(system_4, rhs_4)

    return np.array(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, a23, 0.0],
            [a31, a32, 1.0, 0.0],
            [a41, a42, a43, 1.0],
        ]
    )


def solve_short_run_a_1518(sigma: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Solve the Exercise 15.18 short-run identifying restrictions."""
    sigma = np.asarray(sigma, dtype=float)
    s11, s12, s13 = sigma[0]
    s22, s23 = sigma[1, 1], sigma[1, 2]
    s33 = sigma[2, 2]

    a23 = -s12 / s13
    a32 = -(s23 + a23 * s33) / (s22 + a23 * s23)
    a31 = -(a32 * s12 + s13) / s11

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, a23],
            [a31, a32, 1.0],
        ]
    )


def matrix_frame(matrix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Format a square matrix with matching row and column labels."""
    return pd.DataFrame(matrix, index=labels, columns=labels)


def horizon_frame(horizons: list[int], data: dict[str, list[float]]) -> pd.DataFrame:
    """Build a tidy table of impulse-response values at selected horizons."""
    frame = pd.DataFrame({"h": horizons, **data})
    return frame


def print_title(title: str) -> None:
    """Print a simple exercise header without changing downstream table formats."""
    print(title)
    print("-" * len(title))
