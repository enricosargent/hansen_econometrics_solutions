"""Replicate Exercise 10.31 with clustered resampling.

The code aggregates cross-products by cluster before applying jackknife and
bootstrap formulas, making the clustering unit explicit in the computation.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ch10_utils import (
    bca_interval,
    bootstrap_se,
    format_interval,
    format_number,
    jackknife_se,
    load_xlsx,
)


REPS = 10_000
SEED = 1031


def finite(value: object) -> bool:
    """Check that a workbook cell is a finite numeric value."""
    return isinstance(value, float) and np.isfinite(value)


def build_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare standardized test scores, controls, and school cluster labels."""
    rows = load_xlsx("DDK2011")
    total_scores = np.asarray([row["totalscore"] for row in rows if finite(row["totalscore"])], dtype=float)
    mean_score = float(total_scores.mean())
    sd_score = float(total_scores.std(ddof=1))

    y_values = []
    x_values = []
    groups = []
    for row in rows:
        needed = [
            row["totalscore"],
            row["tracking"],
            row["agetest"],
            row["girl"],
            row["etpteacher"],
            row["percentile"],
            row["schoolid"],
        ]
        if not all(finite(value) for value in needed):
            continue
        y_values.append((float(row["totalscore"]) - mean_score) / sd_score)
        x_values.append(
            [
                float(row["tracking"]),
                float(row["agetest"]),
                float(row["girl"]),
                float(row["etpteacher"]),
                float(row["percentile"]),
                1.0,
            ]
        )
        groups.append(float(row["schoolid"]))
    return np.asarray(y_values), np.asarray(x_values), np.asarray(groups)


def crossproducts_by_cluster(
    y: np.ndarray,
    x: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute each cluster's X'X and X'y contribution."""
    positions: OrderedDict[float, int] = OrderedDict()
    for group in groups:
        positions.setdefault(group, len(positions))
    g = len(positions)
    k = x.shape[1]
    xtx = np.zeros((g, k, k))
    xty = np.zeros((g, k))
    for i, group in enumerate(groups):
        position = positions[group]
        xi = x[i]
        xtx[position] += np.outer(xi, xi)
        xty[position] += xi * y[i]
    return np.asarray(list(positions.keys())), xtx, xty


def solve_crossproduct(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    """Solve beta from aggregated normal equations."""
    return np.linalg.solve(xtx, xty)


def cluster_jackknife_estimates(xtx_by_group: np.ndarray, xty_by_group: np.ndarray) -> np.ndarray:
    """Delete one cluster at a time by subtracting its cross-products."""
    total_xtx = xtx_by_group.sum(axis=0)
    total_xty = xty_by_group.sum(axis=0)
    estimates = []
    for g in range(xtx_by_group.shape[0]):
        estimates.append(solve_crossproduct(total_xtx - xtx_by_group[g], total_xty - xty_by_group[g]))
    return np.asarray(estimates)


def cluster_bootstrap_estimates(
    xtx_by_group: np.ndarray,
    xty_by_group: np.ndarray,
    reps: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap clusters by resampling their precomputed cross-products."""
    rng = np.random.default_rng(seed)
    g = xtx_by_group.shape[0]
    estimates = []
    while len(estimates) < reps:
        sampled = rng.integers(0, g, size=g)
        counts = np.bincount(sampled, minlength=g).astype(float)
        xtx = np.tensordot(counts, xtx_by_group, axes=(0, 0))
        xty = counts @ xty_by_group
        try:
            beta = solve_crossproduct(xtx, xty)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(beta)):
            estimates.append(beta)
    return np.asarray(estimates)


def main() -> None:
    y, x, groups = build_data()
    # Aggregating by school makes the resampling unit match the experimental design.
    group_values, xtx_by_group, xty_by_group = crossproducts_by_cluster(y, x, groups)
    beta = solve_crossproduct(xtx_by_group.sum(axis=0), xty_by_group.sum(axis=0))
    jackknife_estimates = cluster_jackknife_estimates(xtx_by_group, xty_by_group)
    bootstrap_estimates = cluster_bootstrap_estimates(xtx_by_group, xty_by_group, REPS, SEED)

    jackknife_standard_errors = jackknife_se(jackknife_estimates)
    bootstrap_standard_errors = bootstrap_se(bootstrap_estimates)
    names = ["tracking", "age", "girl", "contract_teacher", "percentile", "constant"]

    print("Exercise 10.31")
    print(f"script = python/ch10/ex10_31.py")
    print(f"n = {y.size}, clusters = {group_values.size}, bootstrap replications = {REPS}, seed = {SEED}")
    print()
    print("coefficient, estimate, cluster_bootstrap_se, delete_cluster_jackknife_se, BCa_95")
    for i, name in enumerate(names):
        interval = bca_interval(bootstrap_estimates[:, i], float(beta[i]), jackknife_estimates[:, i])
        print(
            f"{name}, {format_number(beta[i])}, {format_number(bootstrap_standard_errors[i])}, "
            f"{format_number(jackknife_standard_errors[i])}, {format_interval(interval)}"
        )


if __name__ == "__main__":
    main()
