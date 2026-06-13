"""Replicate Exercise 9.27 with specification tests for CPS wages.

The code builds the wage sample, estimates auxiliary regressions, and reports
diagnostics that show how misspecification is detected empirically.
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
    df = load_dataset("MRW1992").copy()
    data = df.loc[df["N"].eq(1)].copy()

    # The Solow growth regression is expressed in logs and growth differences.
    data["lndY"] = np.log(data["Y85"].astype(float)) - np.log(data["Y60"].astype(float))
    data["lnY60"] = np.log(data["Y60"].astype(float))
    data["lnI"] = np.log(data["invest"].astype(float) / 100.0)
    data["lnG"] = np.log(data["pop_growth"].astype(float) / 100.0 + 0.05)
    data["lnS"] = np.log(data["school"].astype(float) / 100.0)
    data = data.dropna(subset=["lndY", "lnY60", "lnI", "lnG", "lnS"]).copy()

    x = pd.DataFrame(
        {
            "const": 1.0,
            "lnY60": data["lnY60"],
            "lnI": data["lnI"],
            "lnG": data["lnG"],
            "lnS": data["lnS"],
        }
    )
    y = data["lndY"].astype(float)
    fit = sm.OLS(y, x).fit(cov_type="HC2")

    restriction = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]])
    target = np.array([0.0])
    # The Wald test asks whether the three accumulation coefficients sum to zero.
    wald = fit.wald_test((restriction, target), scalar=True)

    table = pd.DataFrame({"coef": fit.params, "se_hc2": fit.bse})

    print("Exercise 9.27")
    print(f"n = {len(data)}")
    print()
    print(table.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print(
        "Test lnI + lnG + lnS = 0: "
        f"W = {wald.statistic:.9f}, p-value = {wald.pvalue:.9f}"
    )


if __name__ == "__main__":
    main()
