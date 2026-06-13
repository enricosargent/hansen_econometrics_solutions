"""Shared nonparametric and semiparametric tools for Chapter 20 exercises.

The helpers build polynomial and spline bases, compute model-selection scores,
and save figures so the exercise scripts can focus on empirical comparisons.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PYTHON_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PYTHON_ROOT / "data"


def load_excel(*parts: str) -> pd.DataFrame:
    """Load a Chapter 20 workbook from the local data mirror."""
    return pd.read_excel(DATA_DIR.joinpath(*parts)).copy()


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns before constructing transformations or bases."""
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def rescale_01(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Map a regressor to [0, 1] to stabilize high-order polynomial powers."""
    values = np.asarray(x, dtype=float)
    xmin = float(np.min(values))
    xmax = float(np.max(values))
    return (values - xmin) / (xmax - xmin), xmin, xmax


def polynomial_basis(x: np.ndarray, order: int) -> np.ndarray:
    """Build the intercept-through-order polynomial basis used for series regression."""
    values = np.asarray(x, dtype=float)
    return np.column_stack([values**j for j in range(order + 1)])


def linear_spline_basis(x: np.ndarray, knots: list[float]) -> np.ndarray:
    """Build a continuous piecewise-linear basis with hinge functions at knots."""
    values = np.asarray(x, dtype=float)
    cols = [np.ones_like(values), values]
    for knot in knots:
        cols.append(np.maximum(values - knot, 0.0))
    return np.column_stack(cols)


def quadratic_spline_basis(x: np.ndarray, knots: list[float]) -> np.ndarray:
    """Build a quadratic spline basis with squared positive-part terms at knots."""
    values = np.asarray(x, dtype=float)
    cols = [np.ones_like(values), values, values**2]
    for knot in knots:
        cols.append(np.maximum(values - knot, 0.0) ** 2)
    return np.column_stack(cols)


def ols_fit(y: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit OLS and retain leverage values needed for leave-one-out CV."""
    yvec = np.asarray(y, dtype=float)
    xmat = np.asarray(x, dtype=float)
    result = sm.OLS(yvec, xmat).fit()
    return {
        "beta": np.asarray(result.params, dtype=float),
        "residual": np.asarray(result.resid, dtype=float),
        "hat": np.asarray(result.get_influence().hat_matrix_diag, dtype=float),
        "xtx_inv": np.asarray(result.normalized_cov_params, dtype=float),
        "result": result,
    }


def hc3_covariance(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute HC3 covariance, which inflates residuals by leverage."""
    yvec = np.asarray(y, dtype=float)
    xmat = np.asarray(x, dtype=float)
    return np.asarray(sm.OLS(yvec, xmat).fit(cov_type="HC3").cov_params(), dtype=float)


def cluster_covariance(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Compute a cluster-robust covariance for grouped observations."""
    yvec = np.asarray(y, dtype=float)
    xmat = np.asarray(x, dtype=float)
    groups = np.asarray(groups)
    result = sm.OLS(yvec, xmat).fit()
    robust = result.get_robustcov_results(
        cov_type="cluster",
        groups=groups,
        use_correction=True,
    )
    return np.asarray(robust.cov_params(), dtype=float)


def loocv(y: np.ndarray, x: np.ndarray) -> float:
    """Compute leave-one-out CV without refitting by using OLS leverage values."""
    fit = ols_fit(y, x)
    leave_one_out = fit["residual"] / (1.0 - fit["hat"])
    return float(np.sum(leave_one_out**2))


def aic_gaussian(y: np.ndarray, x: np.ndarray) -> float:
    """Compute the Gaussian AIC score from the residual variance and parameter count."""
    fit = ols_fit(y, x)
    sigma2 = np.mean(fit["residual"] ** 2)
    return float(len(y) * np.log(sigma2) + 2.0 * x.shape[1])


def delete_cluster_cv(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> float:
    """Leave out each cluster to score prediction when observations are grouped."""
    yvec = np.asarray(y, dtype=float)
    xmat = np.asarray(x, dtype=float)
    groups = np.asarray(groups)
    score = 0.0
    for group in np.unique(groups):
        keep = groups != group
        fit = ols_fit(yvec[keep], xmat[keep])
        error = yvec[~keep] - xmat[~keep] @ fit["beta"]
        score += float(error @ error)
    return score


def predict_linear(row: np.ndarray, beta: np.ndarray, covariance: np.ndarray) -> tuple[float, float]:
    """Return a fitted value and its delta-method standard error."""
    point = float(row @ beta)
    variance = float(row @ covariance @ row)
    return point, float(np.sqrt(max(variance, 0.0)))


def confidence_band(prediction: np.ndarray, standard_error: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Form pointwise normal confidence bands around fitted curves."""
    lower = prediction - z * standard_error
    upper = prediction + z * standard_error
    return lower, upper


def save_line_plot(
    filename: str,
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    legend_loc: str = "best",
) -> str:
    """Save fitted curves and optional confidence bands to a temporary PNG."""
    path = Path("/tmp") / filename
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in series:
        ax.plot(x, values, label=label, linewidth=2)
    if lower is not None and upper is not None:
        ax.fill_between(x, lower, upper, alpha=0.2, color="C0", label="95% CI")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def save_scatter_with_line(
    filename: str,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_grid: np.ndarray,
    fit: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> str:
    """Save a scatterplot with a fitted curve and pointwise confidence band."""
    path = Path("/tmp") / filename
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_data, y_data, alpha=0.08, s=8, color="gray")
    ax.plot(x_grid, fit, color="C0", linewidth=2, label="Estimate")
    ax.fill_between(x_grid, lower, upper, color="C0", alpha=0.2, label="95% CI")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def cps09mar_sample() -> pd.DataFrame:
    """Construct the cleaned CPS wage sample used in the series-regression exercises."""
    cps = load_excel("cps09mar", "cps09mar.xlsx")
    cps = ensure_numeric(cps, ["age", "education", "earnings", "hours", "week"])
    cps["wage"] = cps["earnings"] / (cps["hours"] * cps["week"])
    cps["logwage"] = np.log(cps["wage"])
    cps["experience"] = cps["age"] - cps["education"] - 6
    mask = np.isfinite(cps["logwage"]) & (cps["wage"] > 0) & (cps["experience"] >= 0)
    return cps.loc[mask].reset_index(drop=True)


def rr2010_sample() -> pd.DataFrame:
    """Prepare Reinhart-Rogoff growth, debt, and inflation variables."""
    rr = load_excel("RR2010", "RR2010.xlsx")
    rr = ensure_numeric(rr, ["year", "debt", "gdp", "inflation"])
    rr["ylag"] = rr["gdp"].shift(1)
    rr["dlag"] = rr["debt"].shift(1)
    rr["y"] = rr["gdp"]
    return rr.dropna().reset_index(drop=True)


def ddk2011_sample() -> pd.DataFrame:
    """Prepare the school-level tracking experiment sample."""
    ddk = load_excel("DDK2011", "DDK2011.xlsx")
    ddk = ensure_numeric(ddk, ["schoolid", "totalscore", "percentile"])
    ddk["testscore"] = ddk["totalscore"]
    return ddk[["schoolid", "testscore", "percentile"]].dropna().reset_index(drop=True)


def chj2004_sample() -> tuple[pd.DataFrame, list[str]]:
    """Prepare the Philippine transfers sample and its candidate controls."""
    controls = [
        "primary",
        "somesecondary",
        "secondary",
        "someuniversity",
        "university",
        "age",
        "female",
        "married",
        "child1",
        "child7",
        "child15",
        "size",
        "bothwork",
        "notemployed",
        "marriedf",
    ]
    chj = load_excel("CHJ2004", "CHJ2004.xlsx")
    chj = ensure_numeric(chj, ["transfers", "income", *controls])
    chj = chj[["transfers", "income", *controls]].dropna().reset_index(drop=True)
    return chj, controls


def al1999_sample() -> pd.DataFrame:
    """Prepare Angrist-Lavy variables, polynomial controls, and the Maimonides rule."""
    al = load_excel("AL1999", "AL1999.xlsx")
    numeric = ["schlcode", "classize", "disadvantaged", "enrollment", "grade", "avgverb", "avgmath"]
    al = ensure_numeric(al, numeric)
    al["const"] = 1.0
    al["grade4"] = (al["grade"] == 4).astype(float)
    al = al[["const", "schlcode", "classize", "disadvantaged", "enrollment", "grade4", "avgverb", "avgmath"]].dropna().reset_index(drop=True)
    al["c"] = al["classize"] / 40.0
    al["c2"] = al["c"] ** 2
    al["c3"] = al["c"] ** 3
    al["d"] = al["disadvantaged"] / 14.0
    al["d2"] = al["d"] ** 2
    al["d3"] = al["d"] ** 3
    al["cd"] = al["c"] * al["d"]
    al["p"] = al["enrollment"] / (1.0 + np.floor((al["enrollment"] - 1.0) / 40.0))
    al["p1"] = al["p"] / 40.0
    al["p2"] = al["p1"] ** 2
    al["p3"] = al["p1"] ** 3
    al["pd"] = al["p1"] * al["d"]
    return al


def fit_iv_al(depvar: str):
    """Fit the nonparametric-IV control specification for a chosen test score."""
    al = al1999_sample()
    model = IV2SLS(
        dependent=al[depvar],
        exog=al[["const", "d", "d2", "d3", "enrollment", "grade4"]],
        endog=al[["c", "c2", "c3", "cd"]],
        instruments=al[["p1", "p2", "p3", "pd"]],
    )
    result = model.fit(cov_type="clustered", clusters=al["schlcode"])
    return al, result


def print_title(title: str) -> None:
    """Print a simple exercise header without disturbing table formats."""
    print(title)
    print("-" * len(title))
