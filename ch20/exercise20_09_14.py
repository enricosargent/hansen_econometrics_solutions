"""Replicate Exercises 20.9-20.14 with polynomial and spline wage regressions.

The script compares flexible approximations to experience profiles, reports
selection scores, and saves fitted curves with pointwise confidence bands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    aic_gaussian,
    confidence_band,
    cps09mar_sample,
    hc3_covariance,
    loocv,
    ols_fit,
    polynomial_basis,
    predict_linear,
    print_title,
    quadratic_spline_basis,
    rescale_01,
    save_line_plot,
)


def polynomial_summary(data: pd.DataFrame, variable: str, order: int, grid: list[float]) -> pd.DataFrame:
    """Report fitted polynomial values and HC3 SEs at selected regressor points."""
    x = data[variable].to_numpy(dtype=float)
    y = data["logwage"].to_numpy(dtype=float)
    xr, xmin, xmax = rescale_01(x)
    design = polynomial_basis(xr, order)
    fit = ols_fit(y, design)
    covariance = hc3_covariance(y, design)
    rows = []
    for point in grid:
        z = (point - xmin) / (xmax - xmin)
        row = np.array([z**j for j in range(order + 1)], dtype=float)
        estimate, se = predict_linear(row, fit["beta"], covariance)
        rows.append({"x": point, "fit": estimate, "se": se})
    return pd.DataFrame(rows)


def polynomial_curve(data: pd.DataFrame, variable: str, order: int, ngrid: int = 400):
    """Evaluate a fitted polynomial series regression over a dense plotting grid."""
    x = data[variable].to_numpy(dtype=float)
    y = data["logwage"].to_numpy(dtype=float)
    xr, xmin, xmax = rescale_01(x)
    design = polynomial_basis(xr, order)
    fit = ols_fit(y, design)
    covariance = hc3_covariance(y, design)
    grid = np.linspace(xmin, xmax, ngrid)
    z = (grid - xmin) / (xmax - xmin)
    xgrid = polynomial_basis(z, order)
    prediction = xgrid @ fit["beta"]
    standard_error = np.sqrt(np.maximum(np.sum((xgrid @ covariance) * xgrid, axis=1), 0.0))
    return grid, prediction, standard_error


def spline_summary(data: pd.DataFrame, variable: str, knots: list[float], grid: list[float]) -> pd.DataFrame:
    """Report fitted quadratic-spline values and HC3 SEs at selected points."""
    x = data[variable].to_numpy(dtype=float)
    y = data["logwage"].to_numpy(dtype=float)
    design = quadratic_spline_basis(x, knots)
    fit = ols_fit(y, design)
    covariance = hc3_covariance(y, design)
    rows = []
    for point in grid:
        row = np.array([1.0, point, point**2, *[(max(point - knot, 0.0) ** 2) for knot in knots]])
        estimate, se = predict_linear(row, fit["beta"], covariance)
        rows.append({"x": point, "fit": estimate, "se": se})
    return pd.DataFrame(rows)


def spline_curve(data: pd.DataFrame, variable: str, knots: list[float], ngrid: int = 400):
    """Evaluate a fitted quadratic-spline regression over a plotting grid."""
    x = data[variable].to_numpy(dtype=float)
    y = data["logwage"].to_numpy(dtype=float)
    design = quadratic_spline_basis(x, knots)
    fit = ols_fit(y, design)
    covariance = hc3_covariance(y, design)
    grid = np.linspace(float(np.min(x)), float(np.max(x)), ngrid)
    xgrid = quadratic_spline_basis(grid, knots)
    prediction = xgrid @ fit["beta"]
    standard_error = np.sqrt(np.maximum(np.sum((xgrid @ covariance) * xgrid, axis=1), 0.0))
    return grid, prediction, standard_error


def print_selection_table(data: pd.DataFrame, variable: str) -> None:
    """Compare polynomial orders by leave-one-out CV and Gaussian AIC."""
    x = data[variable].to_numpy(dtype=float)
    y = data["logwage"].to_numpy(dtype=float)
    xr, _, _ = rescale_01(x)
    rows = []
    for order in range(1, 9):
        design = polynomial_basis(xr, order)
        rows.append(
            {
                "order": order,
                "CV": loocv(y, design),
                "AIC": aic_gaussian(y, design),
            }
        )
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def print_spline_table(data: pd.DataFrame, variable: str, models: list[tuple[str, list[float]]]) -> None:
    """Compare candidate spline knot sets by CV and AIC."""
    x = data[variable].to_numpy(dtype=float)
    y = data["logwage"].to_numpy(dtype=float)
    rows = []
    for label, knots in models:
        design = quadratic_spline_basis(x, knots)
        rows.append({"model": label, "CV": loocv(y, design), "AIC": aic_gaussian(y, design)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main() -> None:
    cps = cps09mar_sample()
    print_title("Exercise 20.9-20.14")
    print(f"Sample size: {len(cps)}")
    print(f"Experience range: {int(cps['experience'].min())} to {int(cps['experience'].max())}")
    print(f"Education range: {int(cps['education'].min())} to {int(cps['education'].max())}")
    print(f"Observations with experience > 65: {int((cps['experience'] > 65).sum())}")
    print()

    print("Polynomial selection for experience:")
    # Selection tables make the approximation-complexity tradeoff explicit.
    print_selection_table(cps, "experience")
    print()

    print("Polynomial selection for education:")
    print_selection_table(cps, "education")
    print()

    experience_models = [
        ("quadratic", []),
        ("knot20", [20]),
        ("knot20_40", [20, 40]),
        ("knot10_20_30_40", [10, 20, 30, 40]),
    ]
    education_models = [
        ("quadratic", []),
        ("knot10", [10]),
        ("knot5_10_15", [5, 10, 15]),
        ("knot4_8_12_16", [4, 8, 12, 16]),
    ]

    print("Quadratic spline selection for experience:")
    print_spline_table(cps, "experience", experience_models)
    print()

    print("Quadratic spline selection for education:")
    print_spline_table(cps, "education", education_models)
    print()

    summary_209 = polynomial_summary(cps, "experience", 6, [0, 10, 20, 30, 40, 50, 65, 70, 75])
    summary_210 = polynomial_summary(cps, "experience", 5, [0, 10, 20, 30, 40, 50, 65, 70, 75])
    summary_211 = polynomial_summary(cps, "education", 6, [0, 4, 8, 12, 16, 18, 20])
    summary_212 = polynomial_summary(cps, "education", 8, [0, 4, 8, 12, 16, 18, 20])
    summary_213 = spline_summary(cps, "experience", [10, 20, 30, 40], [0, 10, 20, 30, 40, 50, 65, 75])
    summary_214 = spline_summary(cps, "education", [4, 8, 12, 16], [0, 4, 8, 12, 16, 18, 20])

    print("Representative predictions for Exercise 20.9 (poly order 6 on experience):")
    print(summary_209.round(4).to_string(index=False))
    print()

    print("Representative predictions for Exercise 20.10 (poly order 5 on experience):")
    print(summary_210.round(4).to_string(index=False))
    print()

    print("Representative predictions for Exercise 20.11 (poly order 6 on education):")
    print(summary_211.round(4).to_string(index=False))
    print()

    print("Representative predictions for Exercise 20.12 (poly order 8 on education):")
    print(summary_212.round(4).to_string(index=False))
    print()

    print("Representative predictions for Exercise 20.13 (selected spline on experience):")
    print(summary_213.round(4).to_string(index=False))
    print()

    print("Representative predictions for Exercise 20.14 (selected spline on education):")
    print(summary_214.round(4).to_string(index=False))
    print()

    grid, fit, se = polynomial_curve(cps, "experience", 6)
    # Pointwise bands visualize uncertainty in the estimated conditional mean.
    lower, upper = confidence_band(fit, se)
    path = save_line_plot(
        "ch20_ex20_09_poly6_experience.png",
        grid,
        [("Polynomial order 6", fit)],
        xlabel="Experience",
        ylabel="log(wage)",
        title="Exercise 20.9",
        lower=lower,
        upper=upper,
    )
    print(f"Saved plot for Exercise 20.9 to {path}")

    grid, fit, se = polynomial_curve(cps, "experience", 5)
    lower, upper = confidence_band(fit, se)
    path = save_line_plot(
        "ch20_ex20_10_poly5_experience.png",
        grid,
        [("Polynomial order 5", fit)],
        xlabel="Experience",
        ylabel="log(wage)",
        title="Exercise 20.10",
        lower=lower,
        upper=upper,
    )
    print(f"Saved plot for Exercise 20.10 to {path}")

    grid, fit, se = polynomial_curve(cps, "education", 6)
    lower, upper = confidence_band(fit, se)
    path = save_line_plot(
        "ch20_ex20_11_poly6_education.png",
        grid,
        [("Polynomial order 6", fit)],
        xlabel="Education",
        ylabel="log(wage)",
        title="Exercise 20.11",
        lower=lower,
        upper=upper,
    )
    print(f"Saved plot for Exercise 20.11 to {path}")

    grid, fit, se = polynomial_curve(cps, "education", 8)
    lower, upper = confidence_band(fit, se)
    path = save_line_plot(
        "ch20_ex20_12_poly8_education.png",
        grid,
        [("Polynomial order 8", fit)],
        xlabel="Education",
        ylabel="log(wage)",
        title="Exercise 20.12",
        lower=lower,
        upper=upper,
    )
    print(f"Saved plot for Exercise 20.12 to {path}")

    grid = np.linspace(float(cps["experience"].min()), float(cps["experience"].max()), 400)
    series = []
    for label, knots in experience_models:
        _, fit, _ = spline_curve(cps, "experience", knots)
        series.append((label, fit))
    path = save_line_plot(
        "ch20_ex20_13_splines_experience.png",
        grid,
        series,
        xlabel="Experience",
        ylabel="log(wage)",
        title="Exercise 20.13",
    )
    print(f"Saved comparison plot for Exercise 20.13 to {path}")

    grid, fit, se = spline_curve(cps, "experience", [10, 20, 30, 40])
    lower, upper = confidence_band(fit, se)
    path = save_line_plot(
        "ch20_ex20_13_selected_experience.png",
        grid,
        [("Spline with knots 10,20,30,40", fit)],
        xlabel="Experience",
        ylabel="log(wage)",
        title="Exercise 20.13 selected model",
        lower=lower,
        upper=upper,
    )
    print(f"Saved selected-model plot for Exercise 20.13 to {path}")

    grid = np.linspace(float(cps["education"].min()), float(cps["education"].max()), 400)
    series = []
    for label, knots in education_models:
        _, fit, _ = spline_curve(cps, "education", knots)
        series.append((label, fit))
    path = save_line_plot(
        "ch20_ex20_14_splines_education.png",
        grid,
        series,
        xlabel="Education",
        ylabel="log(wage)",
        title="Exercise 20.14",
    )
    print(f"Saved comparison plot for Exercise 20.14 to {path}")

    grid, fit, se = spline_curve(cps, "education", [4, 8, 12, 16])
    lower, upper = confidence_band(fit, se)
    path = save_line_plot(
        "ch20_ex20_14_selected_education.png",
        grid,
        [("Spline with knots 4,8,12,16", fit)],
        xlabel="Education",
        ylabel="log(wage)",
        title="Exercise 20.14 selected model",
        lower=lower,
        upper=upper,
    )
    print(f"Saved selected-model plot for Exercise 20.14 to {path}")


if __name__ == "__main__":
    main()
