"""Replicate Exercise 10.29 with resampling for a growth regression.

The code estimates the convergence parameter and reports resampling intervals
that show how finite-sample uncertainty is summarized.
"""

from __future__ import annotations

import numpy as np

from ch10_utils import (
    bc_interval,
    bootstrap_se,
    delta_se,
    format_interval,
    format_number,
    hc1_covariance,
    jackknife,
    jackknife_se,
    load_xlsx,
    ols,
    pairs_bootstrap,
    percentile_interval,
)


REPS = 10_000
SEED = 1029


def build_data() -> tuple[np.ndarray, np.ndarray]:
    """Prepare the MRW non-oil-country growth-regression sample."""
    rows = load_xlsx("MRW1992")
    values = []
    for row in rows:
        if row["N"] != 1.0:
            continue
        candidate = [row["Y60"], row["Y85"], row["invest"], row["pop_growth"], row["school"]]
        if all(isinstance(value, float) and np.isfinite(value) and value > 0 for value in candidate):
            y60, y85, invest, pop_growth, school = candidate
            values.append(
                [
                    np.log(y85) - np.log(y60),
                    np.log(y60),
                    np.log(invest / 100.0),
                    np.log(pop_growth / 100.0 + 0.05),
                    np.log(school / 100.0),
                ]
            )
    data = np.asarray(values, dtype=float)
    y = data[:, 0]
    x = np.column_stack([np.ones(data.shape[0]), data[:, 1], data[:, 2], data[:, 3], data[:, 4]])
    return y, x


def main() -> None:
    y, x = build_data()
    n = y.size
    names = ["constant", "logY60", "logI", "log(n+g+d)", "logSchool"]
    # Theta aggregates the accumulation terms in the Solow-style regression.
    theta_gradient = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    beta, _, _ = ols(y, x)
    covariance = hc1_covariance(y, x)
    asymptotic_se = np.sqrt(np.diag(covariance))
    theta = float(theta_gradient @ beta)
    theta_asymptotic_se = delta_se(theta_gradient, covariance)

    def estimator(index) -> np.ndarray:
        """Return coefficients plus theta for one bootstrap or jackknife sample."""
        beta_index, _, _ = ols(y[index], x[index])
        return np.r_[beta_index, theta_gradient @ beta_index]

    jackknife_estimates = jackknife(n, estimator)
    bootstrap_estimates = pairs_bootstrap(n, REPS, SEED, estimator)
    jackknife_standard_errors = jackknife_se(jackknife_estimates)
    bootstrap_standard_errors = bootstrap_se(bootstrap_estimates)

    percentile = percentile_interval(bootstrap_estimates[:, -1])
    bc = bc_interval(bootstrap_estimates[:, -1], theta)

    print("Exercise 10.29")
    print(f"script = python/ch10/ex10_29.py")
    print(f"n = {n}, bootstrap replications = {REPS}, seed = {SEED}")
    print()
    print("coefficient, estimate, asymptotic_HC1_se, jackknife_se, bootstrap_se")
    for i, name in enumerate(names):
        print(
            f"{name}, {format_number(beta[i])}, {format_number(asymptotic_se[i])}, "
            f"{format_number(jackknife_standard_errors[i])}, {format_number(bootstrap_standard_errors[i])}"
        )
    print()
    print(f"theta = {format_number(theta)}")
    print(f"theta_asymptotic_HC1_se = {format_number(theta_asymptotic_se)}")
    print(f"theta_jackknife_se = {format_number(jackknife_standard_errors[-1])}")
    print(f"theta_bootstrap_se = {format_number(bootstrap_standard_errors[-1])}")
    print(f"theta_percentile_95 = {format_interval(percentile)}")
    print(f"theta_BC_95 = {format_interval(bc)}")


if __name__ == "__main__":
    main()
