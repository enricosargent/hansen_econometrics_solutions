"""Replicate Chapter 29 machine-learning wage exercises.

The script uses cross-validated Lasso on standardized CPS features and reports
prediction and selected coefficients for separate demographic samples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


EDUCATION_DUMMIES = [12, 13, 14, 15, 16, 18, 20]
MARITAL_DUMMIES = {
    "married": [1, 2, 3],
    "divorced": [5],
    "separated": [6],
    "widowed": [4],
    "never_married": [7],
}
REGION_DUMMIES = {
    "northeast": 1,
    "midwest": 2,
    "south": 3,
    "west": 4,
}


def prepare_cps() -> pd.DataFrame:
    """Prepare the positive-wage CPS sample for Lasso prediction exercises."""
    data = load_dataset("cps09mar").copy()
    data["wage"] = data["earnings"] / (data["hours"] * data["week"])
    data["lwage"] = np.log(data["wage"])
    data["experience"] = data["age"] - data["education"] - 6.0
    return data.loc[np.isfinite(data["lwage"]) & np.isfinite(data["experience"]) & (data["wage"] > 0)].copy()


def design_matrix(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create a broad but interpretable wage-prediction feature set."""
    design = pd.DataFrame(index=sample.index)
    design["education"] = sample["education"].astype(float)
    for value in EDUCATION_DUMMIES:
        design[f"educ_eq_{value}"] = (sample["education"] == value).astype(float)

    experience_scaled = sample["experience"] / 40.0
    for power in range(1, 10):
        design[f"exp40_p{power}"] = experience_scaled**power

    for name, values in MARITAL_DUMMIES.items():
        design[name] = sample["marital"].isin(values).astype(float)

    for name, value in REGION_DUMMIES.items():
        design[f"region_{name}"] = (sample["region"] == value).astype(float)

    design["union"] = sample["union"].astype(float)
    # Drop constant columns after subsetting, since Lasso cannot select them usefully.
    nonconstant = design.std(axis=0) > 0
    return design.loc[:, nonconstant], sample["lwage"].astype(float)


def fit_lasso(sample: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """Fit cross-validated Lasso and convert coefficients back to original units."""
    design, outcome = design_matrix(sample)
    scaler = StandardScaler()
    scaled_design = scaler.fit_transform(design)
    folds = KFold(n_splits=10, shuffle=True, random_state=20260419)
    model = LassoCV(
        alphas=200,
        cv=folds,
        fit_intercept=True,
        max_iter=200_000,
        random_state=20260419,
        tol=1e-6,
    )
    model.fit(scaled_design, outcome)
    # Standardization is only for fitting; reported coefficients use original scales.
    coefficients = model.coef_ / scaler.scale_
    intercept = model.intercept_ - np.dot(scaler.mean_ / scaler.scale_, model.coef_)
    series = pd.Series(coefficients, index=design.columns)
    return float(model.alpha_), float(intercept), series.loc[series.abs() > 1e-8]


def report(name: str, sample: pd.DataFrame) -> None:
    """Print the selected penalty and nonzero Lasso coefficients."""
    alpha, intercept, coefficients = fit_lasso(sample)
    print(f"{name}: n={len(sample)}, alpha={alpha:.12f}, intercept={intercept:.6f}")
    for variable, coefficient in coefficients.items():
        print(f"  {variable:20s} {coefficient: .6f}")
    print()


def main() -> None:
    data = prepare_cps()
    asian_women = data[(data["female"] == 1) & (data["race"] == 4)].copy()
    hispanic_men = data[(data["female"] == 0) & (data["hisp"] == 1)].copy()
    report("Exercise 29.9 Asian women", asian_women)
    report("Exercise 29.10 Hispanic men", hispanic_men)


if __name__ == "__main__":
    main()
