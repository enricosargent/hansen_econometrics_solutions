"""Replicate Exercise 9.28 with robust Wald testing.

The script estimates the wage equation and prints Wald statistics that connect
linear restrictions, robust covariance estimates, and chi-square inference.
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


def main() -> None:
    df = load_dataset("cps09mar").copy()
    # Build the log-wage sample before adding gender, union, and marriage interactions.
    df["wage"] = df["earnings"] / (df["hours"] * df["week"])
    df["lwage"] = np.log(df["wage"])
    df["experience"] = df["age"] - df["education"] - 6
    df["experience_sq_100"] = df["experience"] ** 2 / 100.0

    sample = (
        df["education"].ge(12)
        & df["race"].eq(2)
        & df["hisp"].eq(0)
        & np.isfinite(df["lwage"])
    )
    data = df.loc[sample].dropna(
        subset=["lwage", "education", "experience", "experience_sq_100", "female", "union", "marital"]
    ).copy()

    x = pd.DataFrame(
        {
            "const": 1.0,
            "education": data["education"].astype(float),
            "experience": data["experience"].astype(float),
            "experience_sq_100": data["experience_sq_100"].astype(float),
            "female": data["female"].astype(float),
            "female_union": ((data["female"] == 1) & (data["union"] == 1)).astype(float),
            "male_union": ((data["female"] == 0) & (data["union"] == 1)).astype(float),
            "marriedfemale": (data["female"].eq(1) & data["marital"].isin([1, 2, 3])).astype(float),
            "marriedmale": (data["female"].eq(0) & data["marital"].isin([1, 2, 3])).astype(float),
            "formerlymarriedfemale": (
                data["female"].eq(1) & data["marital"].isin([4, 5, 6])
            ).astype(float),
            "formerlymarriedmale": (
                data["female"].eq(0) & data["marital"].isin([4, 5, 6])
            ).astype(float),
        }
    )
    y = data["lwage"].astype(float)
    fit = sm.OLS(y, x).fit(cov_type="HC2")

    # The four-row restriction matrix tests all marriage-status effects at once.
    restriction = np.zeros((4, x.shape[1]))
    for row, column in enumerate(
        [
            "marriedfemale",
            "marriedmale",
            "formerlymarriedfemale",
            "formerlymarriedmale",
        ]
    ):
        restriction[row, x.columns.get_loc(column)] = 1.0
    wald = fit.wald_test(restriction, scalar=True)

    table = pd.DataFrame({"coef": fit.params, "se_hc2": fit.bse})

    print("Exercise 9.28")
    print(f"n = {len(data)}")
    print()
    print(table.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print(
        "Test marriage-status coefficients equal zero: "
        f"W = {wald.statistic:.9f}, p-value = {wald.pvalue:.12f}"
    )


if __name__ == "__main__":
    main()
