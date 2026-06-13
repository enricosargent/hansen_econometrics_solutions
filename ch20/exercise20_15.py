"""Replicate Exercise 20.15 with a linear-spline transfer regression.

The code fits candidate spline specifications and reports selection criteria
that clarify how knot choices affect the estimated income-transfer profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    confidence_band,
    hc3_covariance,
    linear_spline_basis,
    loocv,
    ols_fit,
    predict_linear,
    print_title,
    rr2010_sample,
    save_line_plot,
)


def model_table(data: pd.DataFrame) -> pd.DataFrame:
    """Score candidate debt spline models by leave-one-out prediction error."""
    rows = []
    y = data["y"].to_numpy(dtype=float)
    ylag = data["ylag"].to_numpy(dtype=float)
    debt = data["dlag"].to_numpy(dtype=float)
    models = [("linear", []), ("knot60", [60]), ("knot40_80", [40, 80])]
    for label, knots in models:
        design = np.column_stack([ylag, linear_spline_basis(debt, knots)])
        rows.append({"model": label, "CV": loocv(y, design)})
    return pd.DataFrame(rows)


def fit_model(data: pd.DataFrame, knots: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Fit one debt spline and return coefficients with HC3 covariance."""
    y = data["y"].to_numpy(dtype=float)
    design = np.column_stack([data["ylag"].to_numpy(dtype=float), linear_spline_basis(data["dlag"].to_numpy(dtype=float), knots)])
    fit = ols_fit(y, design)
    covariance = hc3_covariance(y, design)
    return fit["beta"], covariance


def main() -> None:
    rr = rr2010_sample()
    print_title("Exercise 20.15")
    print(f"Sample: {int(rr['year'].min())} to {int(rr['year'].max())} (n={len(rr)})")
    table = model_table(rr)
    print(table.round(4).to_string(index=False))
    print()

    ylag_mean = float(rr["ylag"].mean())
    # Holding lagged growth at its mean isolates the fitted debt profile.
    for label, knots in [("linear", []), ("knot60", [60]), ("knot40_80", [40, 80])]:
        beta, covariance = fit_model(rr, knots)
        print(f"{label} coefficients:")
        print(np.round(beta, 6))
        for debt in [20, 40, 60, 80, 90, 100, 120]:
            row = np.array([ylag_mean, 1.0, debt, *[max(debt - knot, 0.0) for knot in knots]], dtype=float)
            fit, se = predict_linear(row, beta, covariance)
            m_component = float(np.array([1.0, debt, *[max(debt - knot, 0.0) for knot in knots]]) @ beta[1:])
            print(f"debt={debt:>3}: fit={fit:.4f}, se={se:.4f}, m(debt)={m_component:.4f}")
        print()

    debt_grid = np.linspace(float(rr["dlag"].min()), float(rr["dlag"].max()), 400)
    comparison = []
    # Plot m(D) alone, not the full fitted value, to compare spline shapes.
    for label, knots in [("linear", []), ("knot60", [60]), ("knot40_80", [40, 80])]:
        beta, _ = fit_model(rr, knots)
        component = np.array(
            [np.array([1.0, debt, *[max(debt - knot, 0.0) for knot in knots]]) @ beta[1:] for debt in debt_grid],
            dtype=float,
        )
        comparison.append((label, component))
    path = save_line_plot(
        "ch20_ex20_15_compare.png",
        debt_grid,
        comparison,
        xlabel="Debt/GDP lag",
        ylabel="Estimated m(D)",
        title="Exercise 20.15",
    )
    print(f"Saved comparison plot to {path}")

    beta, covariance = fit_model(rr, [60])
    rows = np.array([[ylag_mean, 1.0, debt, max(debt - 60.0, 0.0)] for debt in debt_grid], dtype=float)
    fit = rows @ beta
    se = np.sqrt(np.maximum(np.sum((rows @ covariance) * rows, axis=1), 0.0))
    lower, upper = confidence_band(fit, se)
    path = save_line_plot(
        "ch20_ex20_15_knot60_ci.png",
        debt_grid,
        [("One-knot spline", fit)],
        xlabel="Debt/GDP lag",
        ylabel="Predicted GDP growth",
        title="Exercise 20.15 one-knot model",
        lower=lower,
        upper=upper,
    )
    print(f"Saved one-knot CI plot to {path}")


if __name__ == "__main__":
    main()
