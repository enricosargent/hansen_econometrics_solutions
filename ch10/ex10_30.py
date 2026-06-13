"""Replicate Exercise 10.30 with a nonlinear wage-hours functional.

The script estimates the functional of OLS coefficients and compares analytic
and simulation-based measures of sampling uncertainty.
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
    quantile,
)


REPS = 10_000
SEED = 1030


def build_data() -> tuple[np.ndarray, np.ndarray]:
    """Prepare the selected CPS wage sample for the nonlinear ratio estimate."""
    rows = load_xlsx("cps09mar")
    values = []
    for row in rows:
        if not (row["race"] == 1.0 and row["hisp"] == 1.0 and row["female"] == 0.0):
            continue
        if not (row["marital"] == 7.0 and row["region"] == 2.0):
            continue
        candidate = [row["earnings"], row["hours"], row["week"], row["education"], row["age"]]
        if not all(isinstance(value, float) and np.isfinite(value) for value in candidate):
            continue
        earnings, hours, week, education, age = candidate
        if earnings <= 0 or hours <= 0 or week <= 0:
            continue
        wage = earnings / (hours * week)
        experience = age - education - 6.0
        values.append([np.log(wage), education, experience, experience**2 / 100.0])
    data = np.asarray(values, dtype=float)
    y = data[:, 0]
    x = np.column_stack([data[:, 1], data[:, 2], data[:, 3], np.ones(data.shape[0])])
    return y, x


def ratio(beta: np.ndarray) -> float:
    """Compute the education-to-experience coefficient ratio."""
    return float(beta[0] / beta[1])


def main() -> None:
    y, x = build_data()
    n = y.size
    beta, _, _ = ols(y, x)
    covariance = hc1_covariance(y, x)
    theta = ratio(beta)
    # The gradient is for beta_education / beta_experience.
    theta_gradient = np.array([1.0 / beta[1], -beta[0] / beta[1] ** 2, 0.0, 0.0])
    theta_asymptotic_se = delta_se(theta_gradient, covariance)

    def estimator(index) -> np.ndarray:
        """Return the ratio estimate for one resampled index vector."""
        beta_index, _, _ = ols(y[index], x[index])
        return np.array([ratio(beta_index)])

    jackknife_estimates = jackknife(n, estimator).reshape(-1)
    bootstrap_estimates = pairs_bootstrap(n, REPS, SEED, estimator).reshape(-1)
    theta_jackknife_se = float(jackknife_se(jackknife_estimates[:, None])[0])
    theta_bootstrap_se = float(bootstrap_se(bootstrap_estimates[:, None])[0])
    bc = bc_interval(bootstrap_estimates, theta)

    print("Exercise 10.30")
    print(f"script = python/ch10/ex10_30.py")
    print(f"n = {n}, bootstrap replications = {REPS}, seed = {SEED}")
    print(f"education_coefficient = {format_number(beta[0])}")
    print(f"experience_coefficient = {format_number(beta[1])}")
    print(f"experience_squared_over_100_coefficient = {format_number(beta[2])}")
    print(f"constant = {format_number(beta[3])}")
    print(f"theta = {format_number(theta)}")
    print(f"theta_asymptotic_HC1_se = {format_number(theta_asymptotic_se)}")
    print(f"theta_jackknife_se = {format_number(theta_jackknife_se)}")
    print(f"theta_bootstrap_se = {format_number(theta_bootstrap_se)}")
    print(
        "theta_bootstrap_quantiles_2.5_50_97.5 = "
        f"({format_number(quantile(bootstrap_estimates, 0.025))}, "
        f"{format_number(quantile(bootstrap_estimates, 0.5))}, "
        f"{format_number(quantile(bootstrap_estimates, 0.975))})"
    )
    print(f"theta_bootstrap_min_max = ({format_number(bootstrap_estimates.min())}, {format_number(bootstrap_estimates.max())})")
    print(f"theta_BC_95 = {format_interval(bc)}")


if __name__ == "__main__":
    main()
