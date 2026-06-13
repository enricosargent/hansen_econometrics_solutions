"""Replicate Exercise 3.26 with alternative wage-regression specifications.

The script prepares the CPS sample, fits the requested OLS models, and reports
the coefficient comparisons that illustrate omitted-variable sensitivity.
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
    # Construct the wage, log-wage, and potential-experience variables first.
    df = load_dataset("cps09mar")
    df = df.copy()
    df["wage"] = df["earnings"] / (df["hours"] * df["week"])
    df["lwage"] = np.log(df["wage"])
    df["experience"] = df["age"] - df["education"] - 6

    sample = df["race"].eq(1) & df["hisp"].eq(1) & df["female"].eq(0)
    data = df.loc[sample].dropna(
        subset=["lwage", "education", "experience", "region", "marital"]
    )

    # Region and marital-status indicators make the omitted categories explicit.
    x = pd.DataFrame(
        {
            "education": data["education"],
            "experience": data["experience"],
            "experience^2/100": data["experience"] ** 2 / 100,
            "Northeast": data["region"].eq(1).astype(float),
            "South": data["region"].eq(3).astype(float),
            "West": data["region"].eq(4).astype(float),
            "married": data["marital"].isin([1, 2, 3]).astype(float),
            "widowed_or_divorced": data["marital"].isin([4, 5]).astype(float),
            "separated": data["marital"].eq(6).astype(float),
            "constant": 1.0,
        }
    )
    fit = sm.OLS(data["lwage"].to_numpy(), x.to_numpy()).fit()

    print("Exercise 3.26")
    print(f"n = {len(data)}")
    print(pd.Series(fit.params, index=x.columns).to_string(float_format=lambda value: f"{value:.9f}"))
    print(f"R2 = {fit.rsquared:.9f}")
    print(f"SSE = {fit.ssr:.9f}")


if __name__ == "__main__":
    main()
