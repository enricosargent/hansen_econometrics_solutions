"""Replicate Exercise 20.17 with polynomial transfer regressions.

The code fits high-order income polynomials, optionally with controls, and saves
estimated curves to compare with Hansen's semiparametric figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    chj2004_sample,
    confidence_band,
    hc3_covariance,
    loocv,
    ols_fit,
    polynomial_basis,
    predict_linear,
    print_title,
    rescale_01,
    save_line_plot,
)


def main() -> None:
    chj, controls = chj2004_sample()
    print_title("Exercise 20.17")
    print(f"Sample size: {len(chj)}")

    y = chj["transfers"].to_numpy(dtype=float)
    income = chj["income"].to_numpy(dtype=float)
    income_scaled, xmin, xmax = rescale_01(income)
    controls_matrix = chj[controls].to_numpy(dtype=float)

    rows = []
    for order in range(1, 9):
        # Income is rescaled before high-order powers to avoid numerical instability.
        design = np.column_stack([polynomial_basis(income_scaled, order), controls_matrix])
        rows.append({"order": order, "CV": loocv(y, design)})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    print()

    selected_order = 6
    design = np.column_stack([polynomial_basis(income_scaled, selected_order), controls_matrix])
    fit = ols_fit(y, design)
    covariance = hc3_covariance(y, design)
    # Predictions hold controls at their sample means to isolate the income profile.
    mean_controls = controls_matrix.mean(axis=0)

    for income_level in [0, 10_000, 20_000, 50_000, 100_000, 150_000, 200_000]:
        z = (income_level - xmin) / (xmax - xmin)
        row = np.concatenate([[z**j for j in range(selected_order + 1)], mean_controls])
        value, se = predict_linear(row, fit["beta"], covariance)
        print(f"income={income_level:>6}: fit={value:.4f}, se={se:.4f}")

    grid = np.linspace(float(np.min(income)), float(np.max(income)), 400)
    zgrid = (grid - xmin) / (xmax - xmin)
    xgrid = np.column_stack([polynomial_basis(zgrid, selected_order), np.tile(mean_controls, (len(grid), 1))])
    prediction = xgrid @ fit["beta"]
    se = np.sqrt(np.maximum(np.sum((xgrid @ covariance) * xgrid, axis=1), 0.0))
    lower, upper = confidence_band(prediction, se)
    path = save_line_plot(
        "ch20_ex20_17_poly_income.png",
        grid,
        [("Polynomial order 6", prediction)],
        xlabel="Income",
        ylabel="Transfers",
        title="Exercise 20.17",
        lower=lower,
        upper=upper,
    )
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
