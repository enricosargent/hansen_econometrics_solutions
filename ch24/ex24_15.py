"""Replicate Exercise 24.15 with clustered quantile-regression inference.

The code bootstraps by school to respect the experimental clustering and prints
quantile treatment-effect estimates with cluster-aware uncertainty.
"""

from __future__ import annotations

import math

import numpy as np

from qr_tools import fit_quantiles, load_xlsx, print_table, quantile_regression


TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]
BOOTSTRAP_REPS = 1000
SEED = 24015


def tracking_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tracked students' scores, percentile regressor, and school clusters."""
    y_values: list[float] = []
    percentile_values: list[float] = []
    school_values: list[int] = []
    for row in load_xlsx("DDK2011"):
        needed = ["totalscore", "percentile", "schoolid"]
        if int(row["tracking"]) != 1 or not all(math.isfinite(float(row[key])) for key in needed):
            continue
        y_values.append(float(row["totalscore"]))
        percentile_values.append(float(row["percentile"]))
        school_values.append(int(row["schoolid"]))
    y = np.asarray(y_values)
    percentile = np.asarray(percentile_values)
    school = np.asarray(school_values)
    x = np.column_stack([np.ones(y.size), percentile])
    return y, x, school


def clustered_bootstrap(
    y: np.ndarray,
    x: np.ndarray,
    school: np.ndarray,
    full_estimates: np.ndarray,
) -> np.ndarray:
    """Resample schools, not students, for quantile-regression standard errors."""
    rng = np.random.default_rng(SEED)
    clusters = np.unique(school)
    groups = [np.flatnonzero(school == cluster) for cluster in clusters]
    estimates = np.empty((BOOTSTRAP_REPS, len(TAUS), x.shape[1]))
    for replication in range(BOOTSTRAP_REPS):
        # Sampling clusters preserves within-school dependence in each bootstrap draw.
        selected = rng.integers(0, len(groups), size=len(groups))
        index = np.concatenate([groups[position] for position in selected])
        y_boot = y[index]
        x_boot = x[index]
        for tau_index, tau in enumerate(TAUS):
            estimates[replication, tau_index] = quantile_regression(
                y_boot,
                x_boot,
                tau,
                full_estimates[tau_index],
            )
    return estimates.std(axis=0, ddof=1)


def main() -> None:
    y, x, school = tracking_sample()
    estimates = fit_quantiles(y, x, TAUS, final_smoothing=1e-7)
    # Bootstrap SEs are computed around the full-sample quantile fits.
    standard_errors = clustered_bootstrap(y, x, school, estimates)
    rows = []
    for tau, beta, se in zip(TAUS, estimates, standard_errors):
        rows.append([tau, beta[0], se[0], beta[1], se[1]])
    print("Exercise 24.15: DDK2011 tracking = 1")
    print(f"n = {y.size}, clusters = {np.unique(school).size}")
    print(f"clustered bootstrap replications = {BOOTSTRAP_REPS}, seed = {SEED}")
    print_table(["tau", "constant", "se_const", "percentile", "se_percentile"], rows)


if __name__ == "__main__":
    main()
