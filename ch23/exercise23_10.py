"""Replicate Exercise 23.10 with a smooth transition regression.

The script estimates the transition parameter and slope coefficients jointly,
showing how a hard threshold can be softened in nonlinear least squares.
"""

import math

import numpy as np
from scipy.optimize import minimize

from common import format_vector, load_workbook_rows, ols_fit, sandwich_covariance


def main():
    rows, index = load_workbook_rows("Nerlove1963/Nerlove1963.xlsx")
    total_cost = np.array([float(row[index["Cost"]]) for row in rows], float)
    output = np.array([float(row[index["output"]]) for row in rows], float)
    price_labor = np.array([float(row[index["Plabor"]]) for row in rows], float)
    price_capital = np.array([float(row[index["Pcapital"]]) for row in rows], float)
    price_fuel = np.array([float(row[index["Pfuel"]]) for row in rows], float)

    log_total_cost = np.log(total_cost)
    log_output = np.log(output)
    input_price_sum = np.log(price_labor) + np.log(price_capital) + np.log(price_fuel)
    sample_size = len(log_total_cost)

    sorted_log_output = np.sort(log_output)
    gamma_lower = float(sorted_log_output[14])
    gamma_upper = float(sorted_log_output[-15])

    def smooth_term(gamma):
        """Construct the smooth-transition interaction at a fixed threshold."""
        logistic = 1.0 / (1.0 + np.exp(-(log_output - gamma)))
        return log_output * logistic

    def fit_given_gamma(gamma):
        """Concentrate out linear coefficients for a fixed transition point."""
        design = np.column_stack(
            [
                np.ones(sample_size),
                log_output,
                input_price_sum,
                smooth_term(gamma),
            ]
        )
        coefficient, residual, sse = ols_fit(design, log_total_cost)
        return coefficient, residual, sse

    def objective(params):
        """Full nonlinear least-squares objective for the global optimizer."""
        beta1, beta2, beta3, beta4, gamma_value = params
        if gamma_value < gamma_lower or gamma_value > gamma_upper:
            return 1e12 + (gamma_value - np.clip(gamma_value, gamma_lower, gamma_upper)) ** 2
        logistic = 1.0 / (1.0 + np.exp(-(log_output - gamma_value)))
        smooth = log_output * logistic
        fitted = beta1 + beta2 * log_output + beta3 * input_price_sum + beta4 * smooth
        residual = log_total_cost - fitted
        return float(residual @ residual)

    best = None
    center = (gamma_lower + gamma_upper) / 2.0
    stages = [
        (0.50, gamma_upper - gamma_lower),
        (0.10, 0.80),
        (0.02, 0.15),
        (0.005, 0.03),
        (0.001, 0.006),
    ]
    for step, radius in stages:
        # The concentrated grid search supplies reliable starts for the optimizer.
        lower = max(gamma_lower, center - radius)
        upper = min(gamma_upper, center + radius)
        grid = np.arange(lower, upper + 1e-12, step)
        for gamma in grid:
            coefficient, residual, sse = fit_given_gamma(gamma)
            if best is None or sse < best["sse"]:
                best = {
                    "gamma": float(gamma),
                    "beta": coefficient,
                    "residual": residual,
                    "sse": sse,
                }
        center = best["gamma"]

    gamma = best["gamma"]
    beta = best["beta"]
    residual = best["residual"]

    global_starts = [
        # Multiple starts guard against local minima in the smooth-transition fit.
        np.append(beta, gamma),
        np.append(beta + np.array([0.2, -0.05, 0.05, -0.05]), gamma_lower),
        np.append(beta + np.array([-0.2, 0.05, -0.05, 0.05]), gamma_upper),
        np.append(beta + np.array([0.1, 0.1, -0.1, 0.1]), (gamma_lower + gamma_upper) / 2.0),
    ]
    global_best = None
    bounds = [(None, None), (None, None), (None, None), (None, None), (gamma_lower, gamma_upper)]
    for start in global_starts:
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            continue
        value = float(result.fun)
        if global_best is None or value < global_best["sse"]:
            global_best = {"params": result.x.copy(), "sse": value}
    if global_best is None:
        raise RuntimeError("Global numerical search failed")

    global_params = global_best["params"]
    global_logistic = 1.0 / (1.0 + np.exp(-(log_output - global_params[4])))
    global_smooth = log_output * global_logistic
    global_residual = log_total_cost - (
        global_params[0]
        + global_params[1] * log_output
        + global_params[2] * input_price_sum
        + global_params[3] * global_smooth
    )

    logistic = 1.0 / (1.0 + np.exp(-(log_output - gamma)))
    smooth = log_output * logistic
    smooth_gamma = -log_output * logistic * (1.0 - logistic)
    jacobian = np.column_stack(
        [
            np.ones(sample_size),
            log_output,
            input_price_sum,
            smooth,
            beta[3] * smooth_gamma,
        ]
    )
    covariance = sandwich_covariance(jacobian, residual)
    standard_errors = np.sqrt(np.diag(covariance))

    print("Exercise 23.10")
    print(f"sample_size = {sample_size}")
    print(f"admissible_gamma_range = [{gamma_lower:.6f}, {gamma_upper:.6f}]")
    print(f"global_search_gamma = {global_params[4]:.6f}")
    print("global_search_parameters = " + format_vector(global_params))
    print(f"global_search_sse = {float(global_residual @ global_residual):.6f}")
    print("std_errors = " + format_vector(np.append(standard_errors[:4], standard_errors[4])))
    print(f"concentrated_search_gamma = {gamma:.6f}")
    print("concentrated_search_parameters = " + format_vector(np.append(beta, gamma)))
    print(f"concentrated_search_sse = {float(residual @ residual):.6f}")


if __name__ == "__main__":
    main()
