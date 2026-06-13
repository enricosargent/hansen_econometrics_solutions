"""Replicate Exercise 15.19 with local projections.

The script estimates horizon-by-horizon regressions after a VAR shock and prints
responses with standard errors, contrasting local projections with VAR IRFs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from common import (
    horizon_frame,
    load_excel,
    log_growth,
    long_run_structural_irf,
    matrix_frame,
    print_title,
    to_numeric,
)


def main() -> None:
    warnings.filterwarnings("ignore")

    df = load_excel("FRED-QD", "FRED-QD.xlsx")
    # Nominal M1 is converted to a real money-growth measure.
    df["nm1"] = to_numeric(df["m1realx"]) * to_numeric(df["cpiaucsl"])
    df["gdp_g"] = log_growth(df["gdpc1"])
    df["m1_g"] = log_growth(df["nm1"])
    data = df[["time", "gdp_g", "m1_g"]].dropna().reset_index(drop=True)

    lagged = {}
    # Four lags match the distributed-lag regression in the exercise.
    for lag in range(1, 5):
        lagged[f"gdp_g_l{lag}"] = data["gdp_g"].shift(lag)
        lagged[f"m1_g_l{lag}"] = data["m1_g"].shift(lag)
    regression_data = pd.concat([data, pd.DataFrame(lagged)], axis=1).dropna().reset_index(drop=True)

    regressors = [f"gdp_g_l{lag}" for lag in range(1, 5)] + [f"m1_g_l{lag}" for lag in range(1, 5)]
    x = add_constant(regression_data[regressors])
    y = regression_data["gdp_g"]
    regression = OLS(y, x).fit(cov_type="HC1")

    joint_restriction = np.zeros((4, x.shape[1]))
    for i, lag in enumerate(range(1, 5)):
        joint_restriction[i, x.columns.get_loc(f"m1_g_l{lag}")] = 1.0
    joint_test = regression.wald_test(joint_restriction)

    # The long-run test sums the money-lag coefficients.
    long_run_restriction = np.zeros((1, x.shape[1]))
    for lag in range(1, 5):
        long_run_restriction[0, x.columns.get_loc(f"m1_g_l{lag}")] = 1.0
    long_run_test = regression.wald_test(long_run_restriction)

    money_rows = []
    for lag in range(1, 5):
        name = f"m1_g_l{lag}"
        money_rows.append(
            {
                "term": name,
                "coef": regression.params[name],
                "se": regression.bse[name],
                "pvalue": regression.pvalues[name],
            }
        )
    money_table = pd.DataFrame(money_rows)

    from statsmodels.tsa.api import VAR

    result = VAR(data[["gdp_g", "m1_g"]]).fit(4)
    _, sirf, _, c_matrix, _ = long_run_structural_irf(result, 20)
    # Cumulative responses are reported as level effects of growth shocks.
    cumulative = sirf.cumsum(axis=0)

    horizons = [0, 1, 2, 4, 8, 12, 20]
    summary = horizon_frame(
        horizons,
        {
            "GDP <- real shock": [cumulative[h, 0, 0] for h in horizons],
            "Money <- real shock": [cumulative[h, 1, 0] for h in horizons],
            "GDP <- monetary shock": [cumulative[h, 0, 1] for h in horizons],
            "Money <- monetary shock": [cumulative[h, 1, 1] for h in horizons],
        },
    )

    print_title("Exercise 15.19")
    print(
        f"Regression sample: {regression_data.time.iloc[0].date()} "
        f"to {regression_data.time.iloc[-1].date()} (n={len(regression_data)})"
    )
    print("Money-lag coefficients:")
    print(money_table.round(4).to_string(index=False))
    print()
    print(
        "Joint test beta_m1,1=...=beta_m1,4=0: "
        f"stat={float(joint_test.statistic):.4f}, p={float(joint_test.pvalue):.4f}"
    )
    print(
        "Long-run test beta_m1,1+...+beta_m1,4=0: "
        f"stat={float(long_run_test.statistic):.4f}, "
        f"p={float(long_run_test.pvalue):.4f}"
    )
    print()
    print("Estimated long-run matrix C:")
    print(matrix_frame(c_matrix, ["gdp_g", "m1_g"]).round(4).to_string())
    print()
    print("Cumulative structural responses (level effects):")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
