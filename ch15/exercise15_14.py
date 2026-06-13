"""Replicate Exercise 15.14 with orthogonalized VAR impulse responses.

The script estimates a macro VAR and prints selected responses to a Cholesky
GDP shock, showing how reduced-form innovations become dynamic effects.
"""

from __future__ import annotations

from common import horizon_frame, load_excel, log_growth, orthogonalized_irf, print_title, to_numeric


def main() -> None:
    df = load_excel("FRED-QD", "FRED-QD.xlsx")
    # Growth rates put GDP and prices on comparable percentage-change scales.
    df["gdp"] = log_growth(df["gdpc1"])
    df["inf"] = log_growth(df["gdpctpi"])
    df["ff"] = to_numeric(df["fedfunds"])
    data = df[["time", "gdp", "inf", "ff"]].dropna().reset_index(drop=True)

    from statsmodels.tsa.api import VAR

    result = VAR(data[["gdp", "inf", "ff"]]).fit(6)
    _, oirf, b = orthogonalized_irf(result, 20)
    # Cumulative sums turn growth-rate responses into level responses.
    shock = oirf[:, :, 0]
    cumulative = shock.cumsum(axis=0)

    horizons = [0, 1, 2, 4, 8, 12, 20]
    summary = horizon_frame(
        horizons,
        {
            "GDP level": [cumulative[h, 0] for h in horizons],
            "Price level": [cumulative[h, 1] for h in horizons],
            "Fed funds": [shock[h, 2] for h in horizons],
        },
    )

    print_title("Exercise 15.14")
    print(f"Sample: {data.time.iloc[0].date()} to {data.time.iloc[-1].date()} (n={len(data)})")
    print("Cholesky impact matrix B:")
    print(b.round(4))
    print()
    print("Responses to the orthogonalized GDP shock:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
