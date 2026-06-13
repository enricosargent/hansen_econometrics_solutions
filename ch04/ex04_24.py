"""Replicate Exercise 4.24 with a multiple-regression wage equation.

The code constructs the CPS design matrix, estimates the model, and prints the
least-squares coefficients and standard errors used in the manual.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


def build_sample() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct Hansen's baseline CPS wage sample and design matrix."""
    df = load_dataset("cps09mar").copy()
    df["wage"] = df["earnings"] / (df["hours"] * df["week"])
    df["lwage"] = np.log(df["wage"])
    df["experience"] = df["age"] - df["education"] - 6
    df["exp2"] = df["experience"] ** 2 / 100

    sample = (
        df["race"].eq(4)
        & df["marital"].eq(7)
        & df["female"].eq(0)
        & (df["experience"] < 45)
    )
    data = df.loc[sample].dropna(subset=["lwage", "education", "experience", "exp2"])
    columns = ["education", "experience", "experience^2/100", "constant"]
    x = np.column_stack(
        [
            data["education"].to_numpy(),
            data["experience"].to_numpy(),
            data["exp2"].to_numpy(),
            np.ones(len(data)),
        ]
    )
    y = data["lwage"].to_numpy()
    return y, x, columns


def main() -> None:
    y, x, columns = build_sample()
    result = sm.OLS(y, x).fit()

    # The same coefficient vector is paired with several covariance estimators.
    standard_errors = {
        "homoskedastic": result.bse,
        "HC0": result.get_robustcov_results(cov_type="HC0").bse,
        "HC1": result.get_robustcov_results(cov_type="HC1").bse,
        "HC2": result.get_robustcov_results(cov_type="HC2").bse,
        "HC3": result.get_robustcov_results(cov_type="HC3").bse,
    }

    table = pd.DataFrame(standard_errors, index=columns)
    print("Exercise 4.24")
    print(f"n = {len(y)}")
    print(table.to_string(float_format=lambda value: f"{value:.9f}"))

    print("\nSecond-language replication")
    print("See javascript/ch04/ex04_24_25.mjs.")


if __name__ == "__main__":
    main()
