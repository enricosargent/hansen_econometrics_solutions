"""Replicate Exercise 9.25 by forming robust confidence intervals.

The code estimates the CPS wage model and converts robust standard errors into
intervals, emphasizing how inference is read from an estimated covariance matrix.
"""

from __future__ import annotations

from pathlib import Path
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


def confidence_intervals(
    coefficients: pd.Series,
    standard_errors: pd.Series,
    level: float = 0.95,
) -> pd.DataFrame:
    """Turn coefficient estimates and robust SEs into Wald confidence intervals."""
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    return pd.DataFrame(
        {
            "coef": coefficients,
            "se_hc2": standard_errors,
            "ci_lower": coefficients - z * standard_errors,
            "ci_upper": coefficients + z * standard_errors,
        }
    )


def main() -> None:
    df = load_dataset("Invest1993").copy()
    data = df.loc[df["year"].eq(1987)].dropna(subset=["inva", "vala", "cfa", "debta"]).copy()

    # The linear model starts with Q, cash flow, and debt as investment predictors.
    x_linear = pd.DataFrame(
        {
            "const": 1.0,
            "Q": data["vala"].astype(float),
            "C": data["cfa"].astype(float),
            "D": data["debta"].astype(float),
        }
    )
    y = data["inva"].astype(float)
    linear_fit = sm.OLS(y, x_linear).fit(cov_type="HC2")

    x_quadratic = x_linear.copy()
    # Quadratic and interaction terms test whether the linear approximation is enough.
    x_quadratic["Q2"] = x_quadratic["Q"] ** 2
    x_quadratic["C2"] = x_quadratic["C"] ** 2
    x_quadratic["D2"] = x_quadratic["D"] ** 2
    x_quadratic["QxC"] = x_quadratic["Q"] * x_quadratic["C"]
    x_quadratic["QxD"] = x_quadratic["Q"] * x_quadratic["D"]
    x_quadratic["CxD"] = x_quadratic["C"] * x_quadratic["D"]
    quadratic_fit = sm.OLS(y, x_quadratic).fit(cov_type="HC2")

    restriction_cd = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # Restriction matrices select the coefficients named in each Wald test.
    restriction_q = np.array([[0.0, 1.0, 0.0, 0.0]])
    restriction_quadratic = np.zeros((6, x_quadratic.shape[1]))
    for row, column in enumerate(["Q2", "C2", "D2", "QxC", "QxD", "CxD"]):
        restriction_quadratic[row, x_quadratic.columns.get_loc(column)] = 1.0

    test_cd = linear_fit.wald_test(restriction_cd, scalar=True)
    test_q = linear_fit.wald_test(restriction_q, scalar=True)
    test_quadratic = quadratic_fit.wald_test(restriction_quadratic, scalar=True)

    linear_table = confidence_intervals(linear_fit.params, linear_fit.bse)

    print("Exercise 9.25")
    print(f"n = {len(data)}")
    print()
    print("Linear specification: I on Q, C, D")
    print(linear_table.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print("Wald tests (HC2)")
    print(f"Test C = D = 0: W = {test_cd.statistic:.9f}, p-value = {test_cd.pvalue:.9f}")
    print(f"Test Q = 0: W = {test_q.statistic:.9f}, p-value = {test_q.pvalue:.9f}")
    print(
        "Quadratic/interactions zero: "
        f"W = {test_quadratic.statistic:.9f}, p-value = {test_quadratic.pvalue:.9f}"
    )


if __name__ == "__main__":
    main()
