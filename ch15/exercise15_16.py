"""Replicate Exercise 15.16 with VAR lag selection and impulse responses.

The code compares lag lengths, estimates the selected VAR, and prints response
summaries that connect information criteria to dynamic interpretation.
"""

from __future__ import annotations

from common import aic_table, horizon_frame, load_excel, log_growth, orthogonalized_irf, print_title, to_numeric


def main() -> None:
    df = load_excel("FRED-MD", "FRED-MD.xlsx")
    # Building permits, housing starts, and real-income growth form the VAR state.
    df["permit"] = to_numeric(df["permit"])
    df["houst"] = to_numeric(df["houst"])
    df["realln_g"] = log_growth(df["realln"])
    data = df[["time", "permit", "houst", "realln_g"]].dropna().reset_index(drop=True)

    series = data[["permit", "houst", "realln_g"]]
    aic = aic_table(series, 8)
    # Lag selection is data-driven but then held fixed for the IRF calculation.
    lag = int(aic.loc[aic["aic"].idxmin(), "lag"])

    from statsmodels.tsa.api import VAR

    result = VAR(series).fit(lag)
    _, oirf, _ = orthogonalized_irf(result, 24)

    horizons = [0, 1, 2, 4, 8, 12, 24]
    summary = horizon_frame(
        horizons,
        {
            "Houst <- permit": [oirf[h, 1, 0] for h in horizons],
            "Houst <- houst": [oirf[h, 1, 1] for h in horizons],
            "Houst <- realln growth": [oirf[h, 1, 2] for h in horizons],
        },
    )

    print_title("Exercise 15.16")
    print(f"Sample: {data.time.iloc[0].date()} to {data.time.iloc[-1].date()} (n={len(data)})")
    print("AIC by lag:")
    print(aic.round(4).to_string(index=False))
    print()
    print(f"Selected lag: {lag}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
