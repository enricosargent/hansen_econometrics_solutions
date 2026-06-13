"""Replicate Exercise 15.15 with an alternative VAR ordering.

The script computes orthogonalized impulse responses after changing the variable
ordering, highlighting the identifying content of Cholesky decompositions.
"""

from __future__ import annotations

from common import horizon_frame, load_excel, orthogonalized_irf, print_title, to_numeric


def main() -> None:
    df = load_excel("Kilian2009", "Kilian2009.xlsx")
    # The sign convention makes the first innovation interpretable as an oil-supply shock.
    df["oil"] = -to_numeric(df["oil"])
    data = df[["time", "oil", "output", "price"]].dropna().reset_index(drop=True)

    from statsmodels.tsa.api import VAR

    result = VAR(data[["oil", "output", "price"]]).fit(4)
    _, oirf, _ = orthogonalized_irf(result, 24)

    # Columns of the Cholesky impact matrix define the three reported shocks.
    horizons = [0, 1, 2, 4, 8, 12, 24]
    summary = horizon_frame(
        horizons,
        {
            "Output <- supply": [oirf[h, 1, 0] for h in horizons],
            "Output <- agg demand": [oirf[h, 1, 1] for h in horizons],
            "Output <- oil demand": [oirf[h, 1, 2] for h in horizons],
        },
    )

    print_title("Exercise 15.15")
    print(f"Sample: {data.time.iloc[0].date()} to {data.time.iloc[-1].date()} (n={len(data)})")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
