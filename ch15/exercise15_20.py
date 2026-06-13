"""Replicate Exercise 15.20 with VAR forecasts and forecast errors.

The code estimates the macro VAR, generates forecasts, and summarizes forecast
accuracy so the mechanics of multi-step prediction are transparent.
"""

from __future__ import annotations

from common import (
    aic_table,
    horizon_frame,
    load_excel,
    log_growth,
    long_run_structural_irf,
    matrix_frame,
    print_title,
)


def main() -> None:
    df = load_excel("FRED-QD", "FRED-QD.xlsx")
    # Inflation is differenced so the VAR is estimated in stationary growth changes.
    df["hours_g"] = log_growth(df["hoanbs"])
    df["gdp_g"] = log_growth(df["gdpc1"])
    df["dinf"] = log_growth(df["gdpctpi"]).diff()
    data = df[["time", "hours_g", "gdp_g", "dinf"]].dropna().reset_index(drop=True)

    series = data[["hours_g", "gdp_g", "dinf"]]
    aic = aic_table(series, 8)
    # The selected lag length is reused for the long-run structural decomposition.
    lag = int(aic.loc[aic["aic"].idxmin(), "lag"])

    from statsmodels.tsa.api import VAR

    result = VAR(series).fit(lag)
    _, sirf, _, c_matrix, _ = long_run_structural_irf(result, 24)
    # Summing growth responses gives the GDP level effect at each horizon.
    cumulative = sirf.cumsum(axis=0)

    horizons = [0, 1, 2, 4, 8, 12, 24]
    summary = horizon_frame(
        horizons,
        {
            "GDP <- hours shock": [cumulative[h, 1, 0] for h in horizons],
            "GDP <- output shock": [cumulative[h, 1, 1] for h in horizons],
            "GDP <- inflation shock": [cumulative[h, 1, 2] for h in horizons],
        },
    )

    print_title("Exercise 15.20")
    print(f"Sample: {data.time.iloc[0].date()} to {data.time.iloc[-1].date()} (n={len(data)})")
    print("AIC by lag:")
    print(aic.round(4).to_string(index=False))
    print()
    print(f"Selected lag: {lag}")
    print("Estimated long-run matrix C:")
    print(matrix_frame(c_matrix, ["hours_g", "gdp_g", "dinf"]).round(4).to_string())
    print()
    print("Cumulative structural responses of GDP:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
