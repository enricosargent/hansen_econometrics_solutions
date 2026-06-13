"""Replicate Chapter 25 binary-choice exercises.

The code estimates logit and probit models for labor-market and marital-status
outcomes, printing marginally interpretable coefficients and standard errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


def load_cps() -> pd.DataFrame:
    """Prepare binary outcomes and nonlinear age terms from the CPS sample."""
    data = load_dataset("cps09mar").copy()
    data["black"] = (data["race"] == 2).astype(float)
    data["union_member"] = (data["union"] == 1).astype(float)
    data["married_123"] = data["marital"].isin([1, 2, 3]).astype(float)
    data["age2_100"] = data["age"] ** 2 / 100.0
    data["age_k40_2_100"] = np.maximum(data["age"] - 40.0, 0.0) ** 2 / 100.0
    return data


def print_table(title: str, result) -> None:
    """Print discrete-choice coefficients and standard errors compactly."""
    print(title)
    table = pd.DataFrame({"coef": result.params, "se": result.bse})
    print(table.to_string(float_format=lambda value: f"{value:.6f}"))
    print()


def union_probit(data: pd.DataFrame, female: int, title: str) -> None:
    """Estimate union-membership probit separately by sex."""
    sample = data.loc[data["female"] == female].dropna(subset=["union_member", "age", "education", "black", "hisp"])
    x = sm.add_constant(sample[["age", "education", "black", "hisp"]])
    result = Probit(sample["union_member"], x).fit(disp=0)
    print(f"{title}: n={len(sample)}")
    print_table("probit coefficients", result)


def college_women_marriage(data: pd.DataFrame) -> None:
    """Estimate age-varying marriage probabilities for college-educated women."""
    sample = data.loc[(data["female"] == 1) & (data["education"] == 16)].dropna(subset=["married_123", "age"])
    x = sm.add_constant(sample[["age", "age2_100", "age_k40_2_100"]])
    result = Logit(sample["married_123"], x).fit(disp=0)
    print(f"Exercise 25.17: college-educated women, n={len(sample)}")
    print_table("logit age-spline coefficients", result)
    ages = np.array([25, 35, 45, 55, 65, 75], dtype=float)
    # The prediction grid translates logit coefficients into age-specific probabilities.
    grid = pd.DataFrame(
        {
            "const": 1.0,
            "age": ages,
            "age2_100": ages**2 / 100.0,
            "age_k40_2_100": np.maximum(ages - 40.0, 0.0) ** 2 / 100.0,
        }
    )
    probabilities = result.predict(grid)
    print(pd.DataFrame({"age": ages, "probability": probabilities}).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()


def marriage_probit(data: pd.DataFrame, female: int, title: str) -> None:
    """Estimate the broader marriage probit with demographics by sex."""
    sample = data.loc[data["female"] == female].dropna(
        subset=["married_123", "age", "education", "black", "hisp"]
    )
    x = sm.add_constant(sample[["age", "age2_100", "age_k40_2_100", "education", "black", "hisp"]])
    result = Probit(sample["married_123"], x).fit(disp=0)
    print(f"{title}: n={len(sample)}")
    print_table("probit coefficients", result)


def main() -> None:
    data = load_cps()
    union_probit(data, female=0, title="Exercise 25.15 men")
    union_probit(data, female=1, title="Exercise 25.16 women")
    college_women_marriage(data)
    marriage_probit(data, female=0, title="Exercise 25.18 men")
    marriage_probit(data, female=1, title="Exercise 25.19 women")


if __name__ == "__main__":
    main()
