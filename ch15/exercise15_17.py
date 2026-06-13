"""Replicate Exercise 15.17 with short-run structural VAR restrictions.

The script solves the imposed contemporaneous matrix and reports structural
impulse responses, showing how restrictions identify economically named shocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    horizon_frame,
    load_excel,
    log_level,
    matrix_frame,
    print_title,
    short_run_structural_irf,
    solve_short_run_a_1517,
    to_numeric,
)


def main() -> None:
    df = load_excel("FRED-QD", "FRED-QD.xlsx")
    # Levels are logged so cumulative structural responses have percentage units.
    df["inv"] = log_level(df["gpdic1"])
    df["price"] = log_level(df["gdpctpi"])
    df["gdp"] = log_level(df["gdpc1"])
    df["ff"] = to_numeric(df["fedfunds"])
    data = df[["time", "inv", "price", "gdp", "ff"]].dropna().reset_index(drop=True)

    exog = pd.DataFrame({"trend": np.arange(len(data), dtype=float)}, index=data.index)

    from statsmodels.tsa.api import VAR

    result = VAR(data[["inv", "price", "gdp", "ff"]], exog=exog).fit(6, trend="c")
    # The A matrix encodes the short-run zero restrictions for this SVAR.
    a_matrix = solve_short_run_a_1517(result.sigma_u)
    _, sirf, _, _ = short_run_structural_irf(result, a_matrix, 20)

    labels = ["inv", "price", "gdp", "ff"]
    horizons = [0, 1, 2, 4, 8, 12, 20]
    summary = horizon_frame(
        horizons,
        {
            "GDP <- FF shock": [sirf[h, 2, 3] for h in horizons],
            "GDP <- GDP shock": [sirf[h, 2, 2] for h in horizons],
            "Price <- GDP shock": [sirf[h, 1, 2] for h in horizons],
        },
    )

    print_title("Exercise 15.17")
    print(f"Sample: {data.time.iloc[0].date()} to {data.time.iloc[-1].date()} (n={len(data)})")
    print("Estimated A matrix:")
    print(matrix_frame(a_matrix, labels).round(4).to_string())
    print()
    print("Selected structural IRFs:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
