"""Replicate Chapter 27 censoring and selection exercises.

The script compares OLS, Tobit maximum likelihood, and CLAD estimates so the
effect of censoring assumptions is visible in the printed tables.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


@dataclass
class TobitResult:
    """Container for Tobit point estimates and likelihood value."""

    beta: np.ndarray
    sigma: float
    loglike: float


def add_constant(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Build a named design matrix with an explicit constant column."""
    design = pd.DataFrame({"constant": np.ones(len(frame))}, index=frame.index)
    for name in names:
        design[name] = frame[name].astype(float)
    return design


def ols_result(outcome: pd.Series, design: pd.DataFrame, cluster: pd.Series | None = None):
    """Fit OLS with HC1 SEs or cluster-robust SEs for comparison models."""
    selected = pd.concat([outcome, design], axis=1).dropna()
    endog = selected.iloc[:, 0].astype(float)
    exog = selected.iloc[:, 1:].astype(float)
    model = sm.OLS(endog, exog)
    if cluster is None:
        return model.fit(cov_type="HC1")
    groups = cluster.loc[selected.index]
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})


def tobit_mle(
    outcome: np.ndarray,
    design: np.ndarray,
    lower: float | None = None,
    upper: float | None = None,
) -> TobitResult:
    """Estimate a censored-normal Tobit model by maximum likelihood."""
    finite = np.isfinite(outcome) & np.all(np.isfinite(design), axis=1)
    outcome = outcome[finite]
    design = design[finite]
    ols_beta = np.asarray(sm.OLS(outcome, design).fit().params, dtype=float)
    residual = outcome - design @ ols_beta
    start = np.r_[ols_beta, np.log(np.std(residual, ddof=design.shape[1]))]

    def negative_loglike(parameter: np.ndarray) -> float:
        """Evaluate uncensored density and censoring probabilities observation by observation."""
        beta = parameter[:-1]
        sigma = np.exp(parameter[-1])
        index = design @ beta
        loglike = np.zeros(outcome.shape[0])
        uncensored = np.ones(outcome.shape[0], dtype=bool)
        if lower is not None:
            lower_censored = outcome <= lower + 1e-10
            loglike[lower_censored] = stats.norm.logcdf((lower - index[lower_censored]) / sigma)
            uncensored &= ~lower_censored
        if upper is not None:
            upper_censored = outcome >= upper - 1e-10
            loglike[upper_censored] = stats.norm.logsf((upper - index[upper_censored]) / sigma)
            uncensored &= ~upper_censored
        standardized = (outcome[uncensored] - index[uncensored]) / sigma
        loglike[uncensored] = stats.norm.logpdf(standardized) - np.log(sigma)
        if not np.all(np.isfinite(loglike)):
            return 1e100
        return float(-np.sum(loglike))

    result = optimize.minimize(negative_loglike, start, method="BFGS", options={"maxiter": 2000})
    if not result.success:
        result = optimize.minimize(negative_loglike, result.x, method="Nelder-Mead", options={"maxiter": 5000})
    beta = result.x[:-1]
    sigma = float(np.exp(result.x[-1]))
    return TobitResult(beta=beta, sigma=sigma, loglike=-float(result.fun))


def clad_estimate(
    outcome: np.ndarray,
    design: np.ndarray,
    lower: float | None = None,
    upper: float | None = None,
    starts: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Estimate Powell's censored least absolute deviations objective."""
    finite = np.isfinite(outcome) & np.all(np.isfinite(design), axis=1)
    outcome = outcome[finite]
    design = design[finite]
    ols_beta = np.asarray(sm.OLS(outcome, design).fit().params, dtype=float)
    start_values = starts or [ols_beta]

    def objective(beta: np.ndarray) -> float:
        """Average absolute deviation after imposing the censoring bound on fitted values."""
        fitted = design @ beta
        if lower is not None:
            fitted = np.maximum(fitted, lower)
        if upper is not None:
            fitted = np.minimum(fitted, upper)
        return float(np.mean(np.abs(outcome - fitted)))

    best = None
    for start in start_values:
        result = optimize.minimize(objective, start, method="Nelder-Mead", options={"maxiter": 5000})
        if best is None or result.fun < best.fun:
            best = result
    return np.asarray(best.x, dtype=float)


def print_linear(label: str, result, names: list[str]) -> None:
    """Print linear-model coefficients in the shared Chapter 27 style."""
    print(label)
    for name in names:
        print(f"  {name:12s} {result.params[name]: .6f}  ({result.bse[name]:.6f})")
    print()


def print_vector(label: str, beta: np.ndarray, names: list[str]) -> None:
    """Print coefficient vectors for Tobit and CLAD estimates."""
    print(label)
    for name, value in zip(names, beta):
        print(f"  {name:12s} {value: .6f}")
    print()


def exercise_27_9() -> None:
    """Compare OLS, Tobit, and CLAD for lower-censored transfer data."""
    data = load_dataset("CHJ2004").copy()
    data["tinkind_k"] = data["tinkind"] / 1000.0
    data["income_k"] = data["income"] / 1000.0
    data["Dincome"] = (data["income_k"] - 1.0) * (data["income_k"] > 1.0)
    design = add_constant(data, ["income_k", "Dincome"])
    outcome = data["tinkind_k"]
    names = list(design.columns)
    full = ols_result(outcome, design)
    print(f"Exercise 27.9: CHJ2004, n={len(data)}")
    print_linear("OLS full sample", full, names)
    censored_percent = float(100.0 * (outcome == 0).mean())
    print(f"Percent censored at zero = {censored_percent:.3f}")
    truncated_mask = outcome > 0
    truncated = ols_result(outcome[truncated_mask], design.loc[truncated_mask])
    print_linear("OLS positive subsample", truncated, names)
    tobit = tobit_mle(outcome.to_numpy(dtype=float), design.to_numpy(dtype=float), lower=0.0)
    print_vector(f"Tobit lower-censored at 0, sigma={tobit.sigma:.6f}, loglike={tobit.loglike:.3f}", tobit.beta, names)
    clad = clad_estimate(
        outcome.to_numpy(dtype=float),
        design.to_numpy(dtype=float),
        lower=0.0,
        starts=[full.params.to_numpy(), truncated.params.to_numpy(), tobit.beta],
    )
    print_vector("CLAD lower-censored at 0", clad, names)


def exercise_27_10() -> None:
    """Compare upper-censoring treatments in a capped log-wage regression."""
    data = load_dataset("cps09mar").copy()
    data["wage"] = data["earnings"] / (data["hours"] * data["week"])
    data["lwage"] = np.log(data["wage"])
    data = data[(data["education"] >= 12) & np.isfinite(data["lwage"]) & (data["wage"] > 0)].copy()
    data["education2"] = data["education"] ** 2
    data["cwage"] = np.minimum(data["lwage"], 3.4)
    design = add_constant(data, ["education", "education2"])
    names = list(design.columns)
    full = ols_result(data["lwage"], design)
    capped = ols_result(data["cwage"], design)
    uncapped_mask = data["cwage"] < 3.4
    truncated = ols_result(data.loc[uncapped_mask, "cwage"], design.loc[uncapped_mask])
    print(f"Exercise 27.10: cps09mar education>=12, n={len(data)}, capped share={(100.0 * (data['lwage'] >= 3.4).mean()):.3f}")
    print_linear("OLS uncapped lwage", full, names)
    print_linear("OLS capped at 3.4", capped, names)
    print_linear("OLS after omitting capped observations", truncated, names)
    tobit = tobit_mle(data["cwage"].to_numpy(dtype=float), design.to_numpy(dtype=float), upper=3.4)
    print_vector(f"Tobit upper-censored at 3.4, sigma={tobit.sigma:.6f}, loglike={tobit.loglike:.3f}", tobit.beta, names)
    clad_outcome = np.minimum(data["lwage"], 3.3)
    clad = clad_estimate(
        clad_outcome.to_numpy(dtype=float),
        design.to_numpy(dtype=float),
        upper=3.3,
        starts=[full.params.to_numpy(), capped.params.to_numpy(), truncated.params.to_numpy(), tobit.beta],
    )
    print_vector("CLAD upper-censored at 3.3", clad, names)


def exercise_27_11() -> None:
    """Compare original and censored test-score regressions with school clustering."""
    data = load_dataset("DDK2011").copy()
    data["testscore"] = (data["totalscore"] - data["totalscore"].mean()) / data["totalscore"].std(ddof=1)
    data["ctest"] = np.maximum(data["testscore"], 0.0)
    data["percentile2"] = data["percentile"] ** 2 / 100.0
    required = ["testscore", "ctest", "tracking", "percentile", "percentile2", "schoolid"]
    data = data.dropna(subset=required).copy()
    design = add_constant(data, ["tracking", "percentile", "percentile2"])
    names = list(design.columns)
    full = ols_result(data["testscore"], design, data["schoolid"])
    censored = ols_result(data["ctest"], design, data["schoolid"])
    positive = data["ctest"] > 0
    truncated = ols_result(data.loc[positive, "ctest"], design.loc[positive], data.loc[positive, "schoolid"])
    print(f"Exercise 27.11: DDK2011, n={len(data)}, schools={data['schoolid'].nunique()}, censored share={(100.0 * (data['ctest'] == 0).mean()):.3f}")
    print_linear("OLS original testscore, school-clustered SE", full, names)
    print_linear("OLS censored ctest, school-clustered SE", censored, names)
    print_linear("OLS positive ctest subsample, school-clustered SE", truncated, names)


def main() -> None:
    exercise_27_9()
    print()
    exercise_27_10()
    print()
    exercise_27_11()


if __name__ == "__main__":
    main()
