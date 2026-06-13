"""Replicate Exercise 20.16 with regression trees as flexible prediction rules.

The script compares tree complexity choices and plots fitted relationships so
the exercise connects nonparametric fit to out-of-sample performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    cluster_covariance,
    confidence_band,
    ddk2011_sample,
    delete_cluster_cv,
    ols_fit,
    predict_linear,
    print_title,
    quadratic_spline_basis,
    save_line_plot,
)


def main() -> None:
    ddk = ddk2011_sample()
    print_title("Exercise 20.16")
    print(f"Sample size: {len(ddk)}")
    print(f"Clusters (schools): {ddk['schoolid'].nunique()}")

    models = [
        ("quadratic", []),
        ("knot50", [50]),
        ("knot33_66", [33, 66]),
        ("knot25_50_75", [25, 50, 75]),
        ("knot20_40_60_80", [20, 40, 60, 80]),
    ]
    y = ddk["testscore"].to_numpy(dtype=float)
    x = ddk["percentile"].to_numpy(dtype=float)
    groups = ddk["schoolid"].to_numpy()

    rows = []
    for label, knots in models:
        design = quadratic_spline_basis(x, knots)
        # Delete-cluster CV evaluates prediction when whole schools are held out.
        rows.append({"model": label, "delete_cluster_CV": delete_cluster_cv(y, design, groups)})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    print()

    grid = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    comparison = []
    for label, knots in models:
        design = quadratic_spline_basis(x, knots)
        fit = ols_fit(y, design)
        xgrid = quadratic_spline_basis(grid, knots)
        comparison.append((label, xgrid @ fit["beta"]))
    path = save_line_plot(
        "ch20_ex20_16_compare.png",
        grid,
        comparison,
        xlabel="Initial percentile",
        ylabel="Test score",
        title="Exercise 20.16",
    )
    print(f"Saved comparison plot to {path}")

    selected_knots = [20, 40, 60, 80]
    design = quadratic_spline_basis(x, selected_knots)
    fit = ols_fit(y, design)
    # Final uncertainty uses school-clustered covariance after selecting the model.
    covariance = cluster_covariance(y, design, groups)
    for percentile in [0, 10, 25, 50, 75, 90, 100]:
        row = np.array([1.0, percentile, percentile**2, *[(max(percentile - knot, 0.0) ** 2) for knot in selected_knots]])
        value, se = predict_linear(row, fit["beta"], covariance)
        print(f"percentile={percentile:>3}: fit={value:.4f}, se={se:.4f}")

    xgrid = quadratic_spline_basis(grid, selected_knots)
    prediction = xgrid @ fit["beta"]
    se = np.sqrt(np.maximum(np.sum((xgrid @ covariance) * xgrid, axis=1), 0.0))
    lower, upper = confidence_band(prediction, se)
    path = save_line_plot(
        "ch20_ex20_16_selected_ci.png",
        grid,
        [("Selected spline", prediction)],
        xlabel="Initial percentile",
        ylabel="Test score",
        title="Exercise 20.16 selected model",
        lower=lower,
        upper=upper,
    )
    print(f"Saved selected-model CI plot to {path}")


if __name__ == "__main__":
    main()
