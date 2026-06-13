"""Replicate Exercise 10.28 with bootstrap inference for a cost function.

The script estimates the target nonlinear functional and compares delta-method,
jackknife, and bootstrap uncertainty estimates.
"""

from __future__ import annotations

import numpy as np

from ch10_utils import (
    bca_interval,
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
SEED = 1028


def build_data() -> tuple[np.ndarray, np.ndarray]:
    """Prepare the log cost function with output and input-price regressors."""
    rows = load_xlsx("Nerlove1963")
    values = []
    for row in rows:
        candidate = [row["Cost"], row["output"], row["Plabor"], row["Pcapital"], row["Pfuel"]]
        if all(isinstance(value, float) and np.isfinite(value) and value > 0 for value in candidate):
            values.append(candidate)
    data = np.asarray(values, dtype=float)
    y = np.log(data[:, 0])
    x = np.column_stack(
        [
            np.ones(data.shape[0]),
            np.log(data[:, 1]),
            np.log(data[:, 2]),
            np.log(data[:, 3]),
            np.log(data[:, 4]),
        ]
    )
    return y, x


def main() -> None:
    y, x = build_data()
    n = y.size
    names = ["constant", "logQ", "logPL", "logPK", "logPF"]
    # Theta is the sum of input-price elasticities, selected by this gradient.
    theta_gradient = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    beta, _, _ = ols(y, x)
    covariance = hc1_covariance(y, x)
    asymptotic_se = np.sqrt(np.diag(covariance))
    theta = float(theta_gradient @ beta)
    theta_asymptotic_se = delta_se(theta_gradient, covariance)

    def estimator(index) -> np.ndarray:
        """Return coefficients plus theta for one resampled index vector."""
        beta_index, _, _ = ols(y[index], x[index])
        return np.r_[beta_index, theta_gradient @ beta_index]

    jackknife_estimates = jackknife(n, estimator)
    bootstrap_estimates = pairs_bootstrap(n, REPS, SEED, estimator)
    jackknife_standard_errors = jackknife_se(jackknife_estimates)
    bootstrap_standard_errors = bootstrap_se(bootstrap_estimates)

    percentile = percentile_interval(bootstrap_estimates[:, -1])
    bca = bca_interval(bootstrap_estimates[:, -1], theta, jackknife_estimates[:, -1])

    print("Exercise 10.28")
    print(f"script = python/ch10/ex10_28.py")
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
    print(f"theta_BCa_95 = {format_interval(bca)}")


if __name__ == "__main__":
    main()
