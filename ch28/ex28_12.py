"""Replicate Exercise 28.12 with model-selection criteria for wage equations.

The code builds Hispanic-women wage specifications and compares AIC, BIC,
cross-validation, and focus-oriented criteria.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


def base_sample() -> pd.DataFrame:
    """Construct the Hispanic-women wage sample for model selection."""
    data = load_dataset("cps09mar").copy()
    data["wage"] = data["earnings"] / (data["hours"] * data["week"])
    data["lwage"] = np.log(data["wage"])
    data["experience"] = data["age"] - data["education"] - 6.0
    sample = data[
        (data["female"] == 1)
        & (data["hisp"] == 1)
        & (data["wage"] > 0)
        & np.isfinite(data["lwage"])
        & np.isfinite(data["experience"])
    ].copy()
    sample["married"] = (sample["marital"] == 1).astype(float)
    return sample


def education_terms(sample: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Return the selected education representation for a candidate model."""
    education = sample["education"].astype(float)
    if kind == "college":
        return pd.DataFrame({"college": (education >= 16).astype(float)}, index=sample.index)
    if kind == "spline":
        return pd.DataFrame(
            {
                "education": education,
                "education_knot9": np.maximum(education - 9.0, 0.0),
            },
            index=sample.index,
        )
    if kind == "dummy":
        values = [12, 13, 14, 16, 18, 20]
        return pd.DataFrame(
            {f"education_{value}": (education == value).astype(float) for value in values},
            index=sample.index,
        )
    raise ValueError(kind)


def experience_terms(sample: pd.DataFrame, degree: int) -> pd.DataFrame:
    """Build polynomial experience terms up to the requested degree."""
    experience = sample["experience"].astype(float)
    return pd.DataFrame(
        {f"experience{power}": experience**power for power in range(1, degree + 1)},
        index=sample.index,
    )


def design_matrix(sample: pd.DataFrame, education_kind: str, experience_degree: int) -> pd.DataFrame:
    """Assemble the wage-regression design for one candidate specification."""
    region_dummies = pd.get_dummies(sample["region"], prefix="region", drop_first=True, dtype=float)
    parts = [
        pd.DataFrame({"constant": np.ones(len(sample)), "married": sample["married"].astype(float)}, index=sample.index),
        region_dummies,
        education_terms(sample, education_kind),
        experience_terms(sample, experience_degree),
    ]
    return pd.concat(parts, axis=1)


def focus_vector(columns: list[str], experience_degree: int) -> np.ndarray:
    """Select the linear combination measuring the return at 30 years' experience."""
    vector = np.zeros(len(columns))
    for power in range(1, experience_degree + 1):
        name = f"experience{power}"
        vector[columns.index(name)] = 30.0**power
    return vector


def loo_cv_sse(result: sm.regression.linear_model.RegressionResultsWrapper) -> float:
    """Compute leave-one-out mean squared error using OLS leverage values."""
    influence = result.get_influence()
    leverage = influence.hat_matrix_diag
    loo_residual = result.resid / (1.0 - leverage)
    return float(np.mean(loo_residual**2))


def model_row(sample: pd.DataFrame, model_number: int, education_kind: str, experience_degree: int, long_focus: float | None) -> dict[str, float | str]:
    """Estimate one candidate model and compute all selection criteria."""
    design = design_matrix(sample, education_kind, experience_degree)
    outcome = sample["lwage"].astype(float)
    result = sm.OLS(outcome, design).fit()
    robust = result.get_robustcov_results(cov_type="HC1")
    columns = list(design.columns)
    focus = focus_vector(columns, experience_degree)
    return_estimate = float(100.0 * (focus @ result.params.to_numpy()))
    robust_cov = np.asarray(robust.cov_params())
    return_se = float(100.0 * np.sqrt(focus @ robust_cov @ focus))
    residual = np.asarray(result.resid)
    sample_size = len(sample)
    regressor_count = design.shape[1]
    sse = float(residual @ residual)
    bic = sample_size * np.log(sse / sample_size) + np.log(sample_size) * regressor_count
    aic = sample_size * np.log(sse / sample_size) + 2.0 * regressor_count
    cv = loo_cv_sse(result)
    if long_focus is None:
        fic = np.nan
    else:
        # FIC targets the squared error of the chosen focus parameter, not global fit.
        fic = sample_size * ((return_estimate - long_focus) / 100.0) ** 2 + 2.0 * sample_size * (return_se / 100.0) ** 2
    return {
        "model": model_number,
        "education": education_kind,
        "experience_degree": experience_degree,
        "k": regressor_count,
        "return_percent": return_estimate,
        "se_percent": return_se,
        "bic": bic,
        "aic": aic,
        "cv": cv,
        "fic": fic,
    }


def main() -> None:
    sample = base_sample()
    specifications = [
        (1, "college", 2),
        (2, "spline", 2),
        (3, "dummy", 2),
        (4, "college", 4),
        (5, "spline", 4),
        (6, "dummy", 4),
        (7, "college", 6),
        (8, "spline", 6),
        (9, "dummy", 6),
    ]
    # The most flexible model supplies the long-model focus used by FIC.
    long_first = model_row(sample, 9, "dummy", 6, None)
    long_focus = float(long_first["return_percent"])
    rows = [model_row(sample, number, education, degree, long_focus) for number, education, degree in specifications]
    result = pd.DataFrame(rows)
    print(f"Exercise 28.12: Hispanic women, n={len(sample)}")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    for criterion in ["bic", "aic", "cv", "fic"]:
        selected = result.loc[result[criterion].idxmin()]
        print(
            f"{criterion.upper()} selects model {int(selected['model'])}: "
            f"return={selected['return_percent']:.3f}%, se={selected['se_percent']:.3f}%"
        )


if __name__ == "__main__":
    main()
