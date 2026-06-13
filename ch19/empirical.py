"""Replicate Chapter 19 nonparametric regression exercises.

The script implements local linear smoothing, bandwidth rules, clustered bands,
and plots so the bias-variance choices behind nonparametrics are visible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MPLCONFIG = ROOT / "python" / "ch19" / ".mplconfig"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIG))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from python.data_loader import load_dataset


FIGURE_DIR = ROOT / "python" / "ch19" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
GAUSSIAN_ROUGHNESS = 1.0 / (2.0 * np.sqrt(np.pi))


def gaussian_kernel(scaled_distance: np.ndarray) -> np.ndarray:
    """Evaluate the Gaussian kernel at bandwidth-scaled distances."""
    return np.exp(-0.5 * scaled_distance**2) / np.sqrt(2.0 * np.pi)


def finite_sample(outcome: np.ndarray, regressor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop missing outcome or regressor values before smoothing."""
    outcome = np.asarray(outcome, dtype=float)
    regressor = np.asarray(regressor, dtype=float)
    finite = np.isfinite(outcome) & np.isfinite(regressor)
    return outcome[finite], regressor[finite]


def fan_gijbels_rot(regressor: np.ndarray, outcome: np.ndarray, lower: float, upper: float) -> float:
    """Compute a Fan-Gijbels-style rule-of-thumb bandwidth."""
    outcome, regressor = finite_sample(outcome, regressor)
    inside = (regressor >= lower) & (regressor <= upper)
    if inside.sum() < 8:
        inside = np.ones_like(regressor, dtype=bool)
    degree = min(4, inside.sum() - 1)
    coefficients = np.polyfit(regressor[inside], outcome[inside], degree)
    fitted = np.polyval(coefficients, regressor[inside])
    residual = outcome[inside] - fitted
    sigma2 = float(np.mean(residual**2))
    # The pilot polynomial estimates curvature, which governs local-linear bias.
    second_derivative = np.polyval(np.polyder(coefficients, 2), regressor[inside])
    curvature = 0.5 * second_derivative
    bias_constant = float(np.mean(curvature**2))
    if not np.isfinite(bias_constant) or bias_constant <= 1e-12:
        return float("inf")
    bandwidth = 0.58 * (sigma2 * (upper - lower) / (inside.sum() * bias_constant)) ** 0.2
    return float(bandwidth)


def local_regression(
    outcome: np.ndarray,
    regressor: np.ndarray,
    grid_values: np.ndarray,
    bandwidth: float,
    method: str,
) -> pd.DataFrame:
    """Estimate Nadaraya-Watson or local-linear curves on a supplied grid."""
    outcome, regressor = finite_sample(outcome, regressor)
    estimates: list[float] = []
    standard_errors: list[float] = []
    for grid_value in grid_values:
        if np.isinf(bandwidth):
            estimate = float(np.mean(outcome))
            standard_error = float(np.std(outcome, ddof=1) / np.sqrt(outcome.size))
        else:
            weights = gaussian_kernel((regressor - grid_value) / bandwidth)
            if method == "nw":
                # NW is a weighted average, so the equivalent weights are direct.
                denominator = weights.sum()
                equivalent_weights = weights / denominator
                estimate = float(equivalent_weights @ outcome)
                residual = outcome - estimate
            elif method == "ll":
                # Local linear regression estimates the intercept at the grid point.
                design = np.column_stack([np.ones_like(regressor), regressor - grid_value])
                cross = design.T @ (weights[:, None] * design)
                cross_inv = np.linalg.pinv(cross)
                beta = cross_inv @ (design.T @ (weights * outcome))
                estimate = float(beta[0])
                equivalent_weights = np.array([1.0, 0.0]) @ cross_inv @ (design.T * weights)
                residual = outcome - design @ beta
            else:
                raise ValueError(method)
            variance = float(np.sum((equivalent_weights**2) * (residual**2)))
            standard_error = np.sqrt(max(variance, 0.0))
        estimates.append(estimate)
        standard_errors.append(standard_error)
    result = pd.DataFrame(
        {
            "grid": grid_values,
            "estimate": estimates,
            "se": standard_errors,
        }
    )
    result["lower"] = result["estimate"] - 1.96 * result["se"]
    result["upper"] = result["estimate"] + 1.96 * result["se"]
    return result


def cluster_local_linear(
    outcome: np.ndarray,
    regressor: np.ndarray,
    cluster: np.ndarray,
    grid_values: np.ndarray,
    bandwidth: float,
) -> pd.DataFrame:
    """Estimate a local-linear curve with cluster-robust pointwise standard errors."""
    outcome = np.asarray(outcome, dtype=float)
    regressor = np.asarray(regressor, dtype=float)
    cluster = np.asarray(cluster)
    finite = np.isfinite(outcome) & np.isfinite(regressor) & pd.notna(cluster)
    outcome = outcome[finite]
    regressor = regressor[finite]
    cluster = cluster[finite]
    fitted_at_observed = local_regression(outcome, regressor, regressor, bandwidth, "ll")["estimate"].to_numpy()
    residual = outcome - fitted_at_observed
    estimates: list[float] = []
    standard_errors: list[float] = []
    for grid_value in grid_values:
        weights = gaussian_kernel((regressor - grid_value) / bandwidth)
        design = np.column_stack([np.ones_like(regressor), regressor - grid_value])
        cross = design.T @ (weights[:, None] * design)
        cross_inv = np.linalg.pinv(cross)
        beta = cross_inv @ (design.T @ (weights * outcome))
        meat = np.zeros((2, 2))
        # Cluster scores preserve the school-level dependence in the tracking data.
        for cluster_value in np.unique(cluster):
            selected = cluster == cluster_value
            cluster_score = design[selected].T @ (weights[selected] * residual[selected])
            meat += np.outer(cluster_score, cluster_score)
        covariance = cross_inv @ meat @ cross_inv
        estimates.append(float(beta[0]))
        standard_errors.append(float(np.sqrt(max(covariance[0, 0], 0.0))))
    result = pd.DataFrame({"grid": grid_values, "estimate": estimates, "se": standard_errors})
    result["lower"] = result["estimate"] - 1.96 * result["se"]
    result["upper"] = result["estimate"] + 1.96 * result["se"]
    return result


def summarize_points(label: str, result: pd.DataFrame, points: list[float]) -> None:
    """Print fitted values nearest to substantively meaningful regressor points."""
    print(label)
    print("point      estimate        se       lower       upper")
    for point in points:
        row = result.iloc[(result["grid"] - point).abs().argmin()]
        print(
            f"{point:6.1f}  {row['estimate']:10.4f}  {row['se']:8.4f}  "
            f"{row['lower']:10.4f}  {row['upper']:10.4f}"
        )
    print()


def save_plot(path: Path, result: pd.DataFrame, title: str, ylabel: str) -> None:
    """Save a nonparametric fit and pointwise confidence band."""
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(result["grid"], result["estimate"], color="black", linewidth=1.6, label="estimate")
    axis.plot(result["grid"], result["lower"], color="black", linewidth=1.0, linestyle="--", label="95% CI")
    axis.plot(result["grid"], result["upper"], color="black", linewidth=1.0, linestyle="--")
    axis.set_title(title)
    axis.set_xlabel("regressor")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def exercise_19_7() -> None:
    """Estimate test-score profiles in the tracking experiment by gender."""
    data = load_dataset("DDK2011")
    lower, upper = 0.0, 100.0
    grid_values = np.linspace(lower, upper, 101)
    for girl_value, group_name in [(1, "girls"), (0, "boys")]:
        sample = data[(data["tracking"] == 1) & (data["girl"] == girl_value)]
        outcome = sample["totalscore"].to_numpy(dtype=float)
        regressor = sample["percentile"].to_numpy(dtype=float)
        cluster = sample["schoolid"].to_numpy()
        bandwidth = fan_gijbels_rot(regressor, outcome, lower, upper)
        result = cluster_local_linear(outcome, regressor, cluster, grid_values, bandwidth)
        print(
            f"Exercise 19.7: {group_name} with tracking, n={len(sample)}, "
            f"clusters={pd.Series(cluster).nunique()}, h_ROT={bandwidth:.4f}"
        )
        summarize_points(f"Local linear test-score regression for {group_name}", result, [10, 25, 50, 75, 90])
        save_plot(FIGURE_DIR / f"ex19_7_{group_name}_tracking.png", result, f"Exercise 19.7 {group_name}", "test score")


def exercise_19_8() -> None:
    """Estimate nonparametric log-wage profiles by experience and sex."""
    data = load_dataset("cps09mar")
    wage = data["earnings"] / (data["hours"] * data["week"])
    experience = data["age"] - data["education"] - 6.0
    sample = data.assign(wage=wage, experience=experience)
    sample = sample[
        (sample["education"] == 20)
        & (sample["experience"].between(0, 40))
        & (sample["wage"] > 0)
        & np.isfinite(sample["wage"])
    ]
    print(f"Exercise 19.8: CPS education=20 and 0<=experience<=40, n={len(sample)}")
    for female_value, group_name in [(0, "men"), (1, "women")]:
        group = sample[sample["female"] == female_value]
        outcome = np.log(group["wage"].to_numpy(dtype=float))
        regressor = group["experience"].to_numpy(dtype=float)
        lower, upper = 0.0, 40.0
        bandwidth = fan_gijbels_rot(regressor, outcome, lower, upper)
        grid_values = np.linspace(lower, upper, 81)
        nw_result = local_regression(outcome, regressor, grid_values, bandwidth, "nw")
        ll_result = local_regression(outcome, regressor, grid_values, bandwidth, "ll")
        print(f"{group_name}: n={len(group)}, h_ROT={bandwidth:.4f}")
        summarize_points(f"NW log-wage profile for {group_name}", nw_result, [0, 10, 20, 30, 40])
        summarize_points(f"LL log-wage profile for {group_name}", ll_result, [0, 10, 20, 30, 40])
        save_plot(FIGURE_DIR / f"ex19_8_{group_name}_nw.png", nw_result, f"Exercise 19.8 NW {group_name}", "log wage")
        save_plot(FIGURE_DIR / f"ex19_8_{group_name}_ll.png", ll_result, f"Exercise 19.8 LL {group_name}", "log wage")


def exercise_19_9() -> None:
    """Estimate investment as a smooth function of Tobin's Q."""
    data = load_dataset("Invest1993")
    sample = data[(data["vala"] <= 5) & np.isfinite(data["vala"]) & np.isfinite(data["inva"])]
    outcome = sample["inva"].to_numpy(dtype=float)
    regressor = sample["vala"].to_numpy(dtype=float)
    lower, upper = float(np.nanmin(regressor)), 5.0
    bandwidth = fan_gijbels_rot(regressor, outcome, lower, upper)
    grid_values = np.linspace(lower, upper, 101)
    nw_result = local_regression(outcome, regressor, grid_values, bandwidth, "nw")
    ll_result = local_regression(outcome, regressor, grid_values, bandwidth, "ll")
    print(f"Exercise 19.9: Invest1993 with Q<=5, n={len(sample)}, h_ROT={bandwidth:.4f}")
    summarize_points("NW investment regression", nw_result, [0.5, 1, 2, 3, 4, 5])
    summarize_points("LL investment regression", ll_result, [0.5, 1, 2, 3, 4, 5])
    save_plot(FIGURE_DIR / "ex19_9_nw.png", nw_result, "Exercise 19.9 NW", "investment")
    save_plot(FIGURE_DIR / "ex19_9_ll.png", ll_result, "Exercise 19.9 LL", "investment")


def exercise_19_10() -> None:
    """Estimate smooth growth relationships with debt and inflation."""
    data = load_dataset("RR2010")
    data = data[np.isfinite(data["debt"]) & np.isfinite(data["gdp"]) & np.isfinite(data["inflation"])]
    debt = data["debt"].to_numpy(dtype=float)
    growth = data["gdp"].to_numpy(dtype=float)
    inflation = data["inflation"].to_numpy(dtype=float)
    debt_bandwidth = fan_gijbels_rot(debt, growth, float(debt.min()), float(debt.max()))
    debt_grid = np.linspace(float(debt.min()), float(debt.max()), 121)
    debt_nw = local_regression(growth, debt, debt_grid, debt_bandwidth, "nw")
    debt_ll = local_regression(growth, debt, debt_grid, debt_bandwidth, "ll")
    inflation_bandwidth = fan_gijbels_rot(inflation, growth, float(np.quantile(inflation, 0.05)), float(np.quantile(inflation, 0.95)))
    inflation_grid = np.linspace(float(np.quantile(inflation, 0.05)), float(np.quantile(inflation, 0.95)), 121)
    inflation_ll = local_regression(growth, inflation, inflation_grid, inflation_bandwidth, "ll")
    print(f"Exercise 19.10: RR2010 United States, n={len(data)}, debt h_ROT={debt_bandwidth:.4f}, inflation h_ROT={inflation_bandwidth:.4f}")
    summarize_points("NW GDP-growth regression on debt", debt_nw, [30, 60, 90, 120])
    summarize_points("LL GDP-growth regression on debt", debt_ll, [30, 60, 90, 120])
    summarize_points("LL GDP-growth regression on inflation", inflation_ll, [-5, 0, 5, 10])
    save_plot(FIGURE_DIR / "ex19_10_debt_nw.png", debt_nw, "Exercise 19.10 NW debt", "GDP growth")
    save_plot(FIGURE_DIR / "ex19_10_debt_ll.png", debt_ll, "Exercise 19.10 LL debt", "GDP growth")
    save_plot(FIGURE_DIR / "ex19_10_inflation_ll.png", inflation_ll, "Exercise 19.10 LL inflation", "GDP growth")


def exercise_19_11() -> None:
    """Estimate a nonlinear autoregression for quarterly GDP growth."""
    data = load_dataset("FRED-QD")
    gdp = data["gdpc1"].astype(float)
    growth = 100.0 * ((gdp / gdp.shift(1)) ** 4 - 1.0)
    lagged_growth = growth.shift(1)
    sample = pd.DataFrame({"growth": growth, "lagged_growth": lagged_growth}).dropna()
    outcome = sample["growth"].to_numpy(dtype=float)
    regressor = sample["lagged_growth"].to_numpy(dtype=float)
    lower = float(np.quantile(regressor, 0.05))
    upper = float(np.quantile(regressor, 0.95))
    bandwidth = fan_gijbels_rot(regressor, outcome, lower, upper)
    grid_values = np.linspace(lower, upper, 101)
    nw_result = local_regression(outcome, regressor, grid_values, bandwidth, "nw")
    ll_result = local_regression(outcome, regressor, grid_values, bandwidth, "ll")
    print(f"Exercise 19.11: FRED-QD GDP growth, n={len(sample)}, h_ROT={bandwidth:.4f}")
    summarize_points("NW nonlinear AR(1) estimate", nw_result, [-4, 0, 2, 4, 6])
    summarize_points("LL nonlinear AR(1) estimate", ll_result, [-4, 0, 2, 4, 6])
    save_plot(FIGURE_DIR / "ex19_11_nw.png", nw_result, "Exercise 19.11 NW", "GDP growth")
    save_plot(FIGURE_DIR / "ex19_11_ll.png", ll_result, "Exercise 19.11 LL", "GDP growth")


def main() -> None:
    exercise_19_7()
    exercise_19_8()
    exercise_19_9()
    exercise_19_10()
    exercise_19_11()


if __name__ == "__main__":
    main()
