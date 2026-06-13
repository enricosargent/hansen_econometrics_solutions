"""Replicate Exercise 23.8 with CES nonlinear least squares.

The script estimates substitution parameters, forms a sandwich covariance, and
prints estimates that connect production-function curvature to inference.
"""

import math

import numpy as np

from common import (
    central_difference_jacobian,
    format_vector,
    load_workbook_rows,
    logsumexp_two,
    ols_fit,
    sandwich_covariance,
    to_float,
)


def load_pss_series(clean_name, dirty_name):
    """Load positive clean/dirty energy inputs and output in logs."""
    rows, index = load_workbook_rows("PSS2017/PSS2017.xlsx")
    countries = []
    log_output = []
    log_clean = []
    log_dirty = []
    for row in rows:
        output_value = to_float(row[index["EG_total"]])
        clean_value = to_float(row[index[clean_name]])
        dirty_value = to_float(row[index[dirty_name]])
        if None in (output_value, clean_value, dirty_value):
            continue
        if min(output_value, clean_value, dirty_value) <= 0:
            continue
        countries.append(row[index["country"]])
        log_output.append(math.log(output_value))
        log_clean.append(math.log(clean_value))
        log_dirty.append(math.log(dirty_value))
    return (
        np.array(countries),
        np.array(log_output, float),
        np.array(log_clean, float),
        np.array(log_dirty, float),
    )


def ces_h(log_clean, log_dirty, rho, alpha):
    """Evaluate the CES aggregator, using Cobb-Douglas as rho approaches zero."""
    if abs(rho) < 1e-10:
        return alpha * log_clean + (1.0 - alpha) * log_dirty
    log_term = logsumexp_two(rho * log_clean, rho * log_dirty, alpha, 1.0 - alpha)
    return log_term / rho


def estimate_ces(log_output, log_clean, log_dirty):
    """Concentrate out linear coefficients while grid-searching CES curvature."""
    best = None
    rho_center = 0.0
    alpha_center = 0.5
    stages = [
        (0.10, 0.05, 0.95, 0.45),
        (0.02, 0.02, 0.15, 0.12),
        (0.005, 0.005, 0.03, 0.03),
        (0.001, 0.001, 0.008, 0.008),
    ]
    for rho_step, alpha_step, rho_radius, alpha_radius in stages:
        # Successively narrower grids give a stable nonlinear least-squares search.
        rho_lower = max(-0.99, rho_center - rho_radius)
        rho_upper = min(0.99, rho_center + rho_radius)
        alpha_lower = max(0.001, alpha_center - alpha_radius)
        alpha_upper = min(0.999, alpha_center + alpha_radius)
        rho_grid = np.arange(rho_lower, rho_upper + 1e-12, rho_step)
        alpha_grid = np.arange(alpha_lower, alpha_upper + 1e-12, alpha_step)
        for rho in rho_grid:
            for alpha in alpha_grid:
                h_value = ces_h(log_clean, log_dirty, rho, alpha)
                design = np.column_stack([np.ones(len(h_value)), h_value])
                coefficient, residual, sse = ols_fit(design, log_output)
                if best is None or sse < best["sse"]:
                    best = {
                        "rho": float(rho),
                        "alpha": float(alpha),
                        "beta": float(coefficient[0]),
                        "nu": float(coefficient[1]),
                        "residual": residual,
                        "sse": sse,
                    }
        rho_center = best["rho"]
        alpha_center = best["alpha"]
    return best


def ces_mean_builder(log_clean, log_dirty):
    """Return the fitted mean function used to compute numerical derivatives."""
    def mean_function(params):
        """Evaluate the CES mean at a candidate nonlinear parameter vector."""
        rho, nu, alpha, beta = params
        return beta + nu * ces_h(log_clean, log_dirty, rho, alpha)

    return mean_function


def summarize(label, result, covariance):
    """Print CES estimates, robust SEs, and the implied substitution elasticity."""
    params = np.array([result["rho"], result["nu"], result["alpha"], result["beta"]], float)
    standard_errors = np.sqrt(np.diag(covariance))
    sigma = 1.0 / (1.0 - result["rho"])
    sigma_se = standard_errors[0] / (1.0 - result["rho"]) ** 2
    print(label)
    print("  parameters =", format_vector(params))
    print("  std_errors =", format_vector(standard_errors))
    print(f"  implied_sigma = {sigma:.6f}")
    print(f"  implied_sigma_se = {sigma_se:.6f}")


def main():
    original_country, original_output, original_clean, original_dirty = load_pss_series("EC_c", "EC_d")
    original_result = estimate_ces(original_output, original_clean, original_dirty)
    original_mean = ces_mean_builder(original_clean, original_dirty)
    original_jacobian = central_difference_jacobian(
        original_mean,
        np.array(
            [
                original_result["rho"],
                original_result["nu"],
                original_result["alpha"],
                original_result["beta"],
            ],
            float,
        ),
        np.array([1e-5, 1e-5, 1e-5, 1e-5], float),
    )
    # Country clustering allows both energy-input equations for a country to move together.
    original_covariance = sandwich_covariance(
        original_jacobian, original_result["residual"], cluster=original_country
    )

    alt_country, alt_output, alt_clean, alt_dirty = load_pss_series("EC_c_alt", "EC_d_alt")
    alt_result = estimate_ces(alt_output, alt_clean, alt_dirty)
    alt_mean = ces_mean_builder(alt_clean, alt_dirty)
    alt_jacobian = central_difference_jacobian(
        alt_mean,
        np.array(
            [alt_result["rho"], alt_result["nu"], alt_result["alpha"], alt_result["beta"]],
            float,
        ),
        np.array([1e-5, 1e-5, 1e-5, 1e-5], float),
    )
    alt_covariance = sandwich_covariance(alt_jacobian, alt_result["residual"], cluster=alt_country)

    print("Exercise 23.8")
    print(f"original_sample_size = {len(original_output)}")
    summarize("original_inputs", original_result, original_covariance)
    print(f"alternative_sample_size = {len(alt_output)}")
    summarize("alternative_inputs", alt_result, alt_covariance)


if __name__ == "__main__":
    main()
