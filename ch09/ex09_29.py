"""Replicate Exercise 9.29 by searching across wage-regression specifications.

The code compares candidate models and reports the resulting fit and inference
quantities so model-selection choices are visible rather than hidden.
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
    # This sample keeps non-Hispanic white and Black workers with at least high school.
    df["wage"] = df["earnings"] / (df["hours"] * df["week"])
    df["lwage"] = np.log(df["wage"])
    df["experience"] = df["age"] - df["education"] - 6
    df["experience_sq_100"] = df["experience"] ** 2 / 100.0

    sample = (
        df["education"].ge(12)
        & df["race"].isin([1, 2])
        & df["hisp"].eq(0)
        & np.isfinite(df["lwage"])
    )
    data = df.loc[sample].dropna(
        subset=["lwage", "education", "experience", "experience_sq_100", "female", "union", "marital"]
    ).copy()

    white_male = (data["race"] == 1) & (data["female"] == 0)
    white_female = (data["race"] == 1) & (data["female"] == 1)
    black_male = (data["race"] == 2) & (data["female"] == 0)
    black_female = (data["race"] == 2) & (data["female"] == 1)

    # Education interactions let the return to schooling differ by race-sex group.
    x = pd.DataFrame(
        {
            "const": 1.0,
            "education": data["education"].astype(float),
            "experience": data["experience"].astype(float),
            "experience_sq_100": data["experience_sq_100"].astype(float),
            "female": data["female"].astype(float),
            "black": data["race"].eq(2).astype(float),
            "blackfemale": black_female.astype(float),
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
            "education_whitefemale": data["education"].astype(float) * white_female.astype(float),
            "education_blackmale": data["education"].astype(float) * black_male.astype(float),
            "education_blackfemale": data["education"].astype(float) * black_female.astype(float),
        }
    )
    y = data["lwage"].astype(float)
    fit = sm.OLS(y, x).fit(cov_type="HC2")

    restriction = np.zeros((3, x.shape[1]))
    for row, column in enumerate(
        ["education_whitefemale", "education_blackmale", "education_blackfemale"]
    ):
        restriction[row, x.columns.get_loc(column)] = 1.0
    wald = fit.wald_test(restriction, scalar=True)

    base = float(fit.params["education"])
    # Implied returns translate interaction coefficients back into group slopes.
    implied_returns = pd.Series(
        {
            "white_male": base,
            "white_female": base + float(fit.params["education_whitefemale"]),
            "black_male": base + float(fit.params["education_blackmale"]),
            "black_female": base + float(fit.params["education_blackfemale"]),
        }
    )

    table = pd.DataFrame({"coef": fit.params, "se_hc2": fit.bse})

    print("Exercise 9.29")
    print(f"n = {len(data)}")
    print()
    print(table.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print("Implied education returns by group")
    print(implied_returns.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print(
        "Test of common education return across the four groups: "
        f"W = {wald.statistic:.9f}, p-value = {wald.pvalue:.9f}"
    )


if __name__ == "__main__":
    main()
