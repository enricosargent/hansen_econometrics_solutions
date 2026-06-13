"""Replicate Exercise 23.9 with threshold regression.

The code searches over thresholds, refits the regression at each candidate, and
reports the selected kink that minimizes the nonlinear least-squares criterion.
"""

import numpy as np

from common import format_vector, load_workbook_rows, ols_fit, sandwich_covariance


def negative_part(values):
    """Return the left-threshold basis term."""
    return np.minimum(values, 0.0)


def positive_part(values):
    """Return the right-threshold basis term."""
    return np.maximum(values, 0.0)


def main():
    rows, index = load_workbook_rows("RR2010/RR2010.xlsx")
    debt = np.array([float(row[index["debt"]]) for row in rows], float)
    inflation = np.array([float(row[index["inflation"]]) for row in rows], float)

    outcome = inflation[1:]
    debt_ratio = debt[1:]
    lagged_inflation = inflation[:-1]
    sample_size = len(outcome)

    best = None
    center = float(debt_ratio.mean())
    stages = [
        (5.0, float(debt_ratio.max() - debt_ratio.min())),
        (1.0, 15.0),
        (0.2, 3.0),
        (0.05, 0.6),
        (0.01, 0.12),
    ]

    def fit_given_threshold(threshold):
        """Fit the linear pieces after fixing the threshold."""
        design = np.column_stack(
            [
                negative_part(debt_ratio - threshold),
                positive_part(debt_ratio - threshold),
                lagged_inflation,
                np.ones(sample_size),
            ]
        )
        coefficient, residual, sse = ols_fit(design, outcome)
        return coefficient, residual, sse

    for step, radius in stages:
        # Coarse-to-fine grids avoid making the threshold search depend on one mesh.
        lower = max(float(debt_ratio.min()) + 1e-6, center - radius)
        upper = min(float(debt_ratio.max()) - 1e-6, center + radius)
        grid = np.arange(lower, upper + 1e-12, step)
        for threshold in grid:
            coefficient, residual, sse = fit_given_threshold(threshold)
            if best is None or sse < best["sse"]:
                best = {
                    "c": float(threshold),
                    "beta": coefficient,
                    "residual": residual,
                    "sse": sse,
                }
        center = best["c"]

    threshold = best["c"]
    beta = best["beta"]
    residual = best["residual"]

    jacobian = np.column_stack(
        # The final column is the derivative of the fitted value with respect to c.
        [
            negative_part(debt_ratio - threshold),
            positive_part(debt_ratio - threshold),
            lagged_inflation,
            np.ones(sample_size),
            -beta[0] * (debt_ratio < threshold) - beta[1] * (debt_ratio > threshold),
        ]
    )
    covariance = sandwich_covariance(jacobian, residual)
    standard_errors = np.sqrt(np.diag(covariance))

    print("Exercise 23.9")
    print(f"sample_size = {sample_size}")
    print(f"threshold_c = {threshold:.6f}")
    print("parameters = " + format_vector(np.append(beta, threshold)))
    print("std_errors = " + format_vector(np.append(standard_errors[:4], standard_errors[4])))
    print(f"below_threshold = {int(np.sum(debt_ratio < threshold))}")
    print(f"above_threshold = {int(np.sum(debt_ratio > threshold))}")


if __name__ == "__main__":
    main()
