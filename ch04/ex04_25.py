"""Replicate Exercise 4.25 by comparing nested CPS wage regressions.

The script assembles the sample, estimates the unrestricted and restricted
models, and reports the statistics that connect regression output to inference.
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
    """Construct the richer CPS wage design with region and marital indicators."""
    df = load_dataset("cps09mar").copy()
    df["wage"] = df["earnings"] / (df["hours"] * df["week"])
    df["lwage"] = np.log(df["wage"])
    df["experience"] = df["age"] - df["education"] - 6

    sample = df["race"].eq(1) & df["hisp"].eq(1) & df["female"].eq(0)
    data = df.loc[sample].dropna(
        subset=["lwage", "education", "experience", "region", "marital"]
    )
    columns = [
        "education",
        "experience",
        "experience^2/100",
        "Northeast",
        "South",
        "West",
        "married",
        "widowed_or_divorced",
        "separated",
        "constant",
    ]
    x = np.column_stack(
        [
            data["education"].to_numpy(),
            data["experience"].to_numpy(),
            (data["experience"] ** 2 / 100).to_numpy(),
            data["region"].eq(1).astype(float).to_numpy(),
            data["region"].eq(3).astype(float).to_numpy(),
            data["region"].eq(4).astype(float).to_numpy(),
            data["marital"].isin([1, 2, 3]).astype(float).to_numpy(),
            data["marital"].isin([4, 5]).astype(float).to_numpy(),
            data["marital"].eq(6).astype(float).to_numpy(),
            np.ones(len(data)),
        ]
    )
    y = data["lwage"].to_numpy()
    return y, x, columns


def main() -> None:
    y, x, columns = build_sample()
    result = sm.OLS(y, x).fit()
    # HC3 emphasizes the leverage adjustment in finite samples.
    standard_errors = result.get_robustcov_results(cov_type="HC3").bse

    print("Exercise 4.25")
    print(f"n = {len(y)}")
    print(pd.Series(standard_errors, index=columns).to_string(float_format=lambda value: f"{value:.9f}"))
    print("\nSecond-language replication")
    print("See javascript/ch04/ex04_24_25.mjs.")


if __name__ == "__main__":
    main()
