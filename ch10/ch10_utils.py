"""Shared numerical tools for Chapter 10 resampling exercises.

The helpers keep OLS, delta-method, jackknife, bootstrap, and interval formulas
in one place so the exercise scripts can focus on the econometric interpretation.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import statsmodels.api as sm


PYTHON_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PYTHON_ROOT / "data"
NORMAL = NormalDist()


def as_float(value: object) -> float:
    """Convert spreadsheet cells to floats while treating blanks as missing."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip()
        if text in {"", ".", "NA"}:
            return float("nan")
        return float(text)
    return float(value)


def load_xlsx(name: str) -> list[dict[str, float | str]]:
    """Read a Chapter 10 workbook into row dictionaries for transparent parsing."""
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


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return OLS coefficients, residuals, and the design cross-product."""
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    result = sm.OLS(y, x).fit()
    beta = np.asarray(result.params, dtype=float)
    residual = np.asarray(result.resid, dtype=float)
    xtx = x.T @ x
    return beta, residual, xtx


def hc1_covariance(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute the HC1 sandwich covariance used for robust delta-method inference."""
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    return np.asarray(sm.OLS(y, x).fit(cov_type="HC1").cov_params(), dtype=float)


def delta_se(gradient: np.ndarray, covariance: np.ndarray) -> float:
    """Apply the delta method to a scalar smooth function of regression estimates."""
    variance = float(gradient @ covariance @ gradient)
    return float(np.sqrt(max(variance, 0.0)))


def jackknife(n: int, estimator) -> np.ndarray:
    """Recompute an estimator after deleting each observation once."""
    estimates = []
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        estimates.append(np.asarray(estimator(mask), dtype=float))
        mask[i] = True
    return np.asarray(estimates)


def jackknife_se(estimates: np.ndarray) -> np.ndarray:
    """Convert leave-one-out estimates into the usual jackknife standard error."""
    estimates = np.asarray(estimates, dtype=float)
    n = estimates.shape[0]
    centered = estimates - estimates.mean(axis=0)
    return np.sqrt(((n - 1.0) / n) * np.sum(centered**2, axis=0))


def pairs_bootstrap(n: int, reps: int, seed: int, estimator) -> np.ndarray:
    """Draw observation-index bootstrap samples and evaluate the estimator."""
    rng = np.random.default_rng(seed)
    estimates = []
    while len(estimates) < reps:
        index = rng.integers(0, n, size=n)
        try:
            estimate = np.asarray(estimator(index), dtype=float)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(estimate)):
            estimates.append(estimate)
    return np.asarray(estimates)


def bootstrap_se(estimates: np.ndarray) -> np.ndarray:
    """Summarize bootstrap draws by their sample standard deviation."""
    return np.asarray(estimates, dtype=float).std(axis=0, ddof=1)


def quantile(values: np.ndarray, probability: float) -> float:
    """Use NumPy's linear empirical quantile for bootstrap intervals."""
    return float(np.quantile(values, probability, method="linear"))


def percentile_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Form a percentile bootstrap confidence interval from sorted draws."""
    return (
        quantile(values, alpha / 2.0),
        quantile(values, 1.0 - alpha / 2.0),
    )


def bc_interval(
    values: np.ndarray,
    estimate: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Form Efron's bias-corrected bootstrap interval for a scalar estimate."""
    values = np.asarray(values, dtype=float)
    proportion = (
        np.sum(values < estimate) + 0.5 * np.sum(values == estimate)
    ) / values.size
    proportion = min(max(float(proportion), 1.0 / (2.0 * values.size)), 1.0 - 1.0 / (2.0 * values.size))
    z0 = NORMAL.inv_cdf(proportion)
    lower_p = NORMAL.cdf(2.0 * z0 + NORMAL.inv_cdf(alpha / 2.0))
    upper_p = NORMAL.cdf(2.0 * z0 + NORMAL.inv_cdf(1.0 - alpha / 2.0))
    return quantile(values, lower_p), quantile(values, upper_p)


def bca_interval(
    values: np.ndarray,
    estimate: float,
    jackknife_estimates: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Form the BCa interval, adding jackknife acceleration to bias correction."""
    values = np.asarray(values, dtype=float)
    jackknife_estimates = np.asarray(jackknife_estimates, dtype=float)
    proportion = (
        np.sum(values < estimate) + 0.5 * np.sum(values == estimate)
    ) / values.size
    proportion = min(max(float(proportion), 1.0 / (2.0 * values.size)), 1.0 - 1.0 / (2.0 * values.size))
    z0 = NORMAL.inv_cdf(proportion)

    mean_jackknife = float(jackknife_estimates.mean())
    influence = mean_jackknife - jackknife_estimates
    denominator = 6.0 * float(np.sum(influence**2)) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.sum(influence**3)) / denominator

    def adjusted(probability: float) -> float:
        """Map nominal tail probabilities through the BCa correction."""
        z = NORMAL.inv_cdf(probability)
        numerator = z0 + z
        return NORMAL.cdf(z0 + numerator / (1.0 - acceleration * numerator))

    return (
        quantile(values, adjusted(alpha / 2.0)),
        quantile(values, adjusted(1.0 - alpha / 2.0)),
    )


def format_number(value: float) -> str:
    """Format scalars with the precision used in Chapter 10 tables."""
    return f"{float(value):.9f}"


def format_interval(interval: tuple[float, float]) -> str:
    """Format a two-endpoint interval for printed output."""
    return f"({format_number(interval[0])}, {format_number(interval[1])})"
