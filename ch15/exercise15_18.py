"""Replicate Exercise 15.18 with a second short-run SVAR identification.

The code solves the exercise-specific contemporaneous restrictions and prints
the resulting structural responses for comparison with the recursive case.
"""

from __future__ import annotations

from common import (
    horizon_frame,
    load_excel,
    matrix_frame,
    print_title,
    short_run_structural_irf,
    solve_short_run_a_1518,
    to_numeric,
)


def main() -> None:
    df = load_excel("Kilian2009", "Kilian2009.xlsx")
    # Reversing oil production preserves the supply-shock interpretation.
    df["oil"] = -to_numeric(df["oil"])
    data = df[["time", "oil", "output", "price"]].dropna().reset_index(drop=True)

    from statsmodels.tsa.api import VAR

    result = VAR(data[["oil", "output", "price"]]).fit(4)
    # This alternative A matrix implements the exercise's nonrecursive restrictions.
    a_matrix = solve_short_run_a_1518(result.sigma_u)
    _, sirf, _, _ = short_run_structural_irf(result, a_matrix, 24)

    labels = ["oil", "output", "price"]
    horizons = [0, 1, 2, 4, 8, 12, 24]
    summary = horizon_frame(
        horizons,
        {
            "Price <- oil supply": [sirf[h, 2, 0] for h in horizons],
            "Price <- agg demand": [sirf[h, 2, 1] for h in horizons],
            "Price <- oil demand": [sirf[h, 2, 2] for h in horizons],
        },
    )

    print_title("Exercise 15.18")
    print(f"Sample: {data.time.iloc[0].date()} to {data.time.iloc[-1].date()} (n={len(data)})")
    print("Estimated A matrix:")
    print(matrix_frame(a_matrix, labels).round(4).to_string())
    print()
    print("Selected structural IRFs:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
