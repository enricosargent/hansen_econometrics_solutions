"""Replicate Exercise 9.24 with heteroskedasticity-robust wage inference.

The script fits the requested OLS specification and reports robust covariance
calculations that connect asymptotic theory to regression output.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    rng = np.random.default_rng(12345)

    # The simulation compares inference for beta with inference for exp(beta).
    alpha = 0.0
    beta_true = 1.0
    theta_true = float(np.exp(beta_true))
    n = 50
    b = 1000

    rows: list[dict[str, float]] = []
    for _ in range(b):
        x = rng.uniform(0.0, 1.0, n)
        e = rng.normal(0.0, 1.0, n)
        y = alpha + beta_true * x + e

        design = np.column_stack([np.ones(n), x])
        fit = sm.OLS(y, design).fit()
        robust = fit.get_robustcov_results(cov_type="HC1")
        covariance = robust.cov_params()

        beta_hat = float(fit.params[1])
        se_beta = float(robust.bse[1])

        theta_hat = float(np.exp(beta_hat))
        # Delta-method standard errors use the gradient of exp(beta).
        gradient = np.array([0.0, theta_hat])
        se_theta = float(np.sqrt(gradient @ covariance @ gradient))

        rows.append(
            {
                "beta_hat": beta_hat,
                "theta_hat": theta_hat,
                "t_beta": (beta_hat - beta_true) / se_beta,
                "t_theta": (theta_hat - theta_true) / se_theta,
            }
        )

    results = pd.DataFrame(rows)

    # Rejection frequencies and quantiles diagnose the normal approximation.
    summary = pd.Series(
        {
            "B": float(b),
            "n": float(n),
            "mean_beta_hat": float(results["beta_hat"].mean()),
            "bias_beta_hat": float(results["beta_hat"].mean() - beta_true),
            "mean_theta_hat": float(results["theta_hat"].mean()),
            "bias_theta_hat": float(results["theta_hat"].mean() - theta_true),
            "prob_t_beta_gt_1.645": float((results["t_beta"] > 1.645).mean()),
            "prob_t_theta_gt_1.645": float((results["t_theta"] > 1.645).mean()),
            "sd_t_beta": float(results["t_beta"].std(ddof=1)),
            "sd_t_theta": float(results["t_theta"].std(ddof=1)),
        }
    )

    print("Exercise 9.24")
    print(summary.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print("Quantiles of T_beta")
    print(
        results["t_beta"]
        .quantile([0.025, 0.5, 0.975])
        .to_string(float_format=lambda value: f"{value:.9f}")
    )
    print()
    print("Quantiles of T_theta")
    print(
        results["t_theta"]
        .quantile([0.025, 0.5, 0.975])
        .to_string(float_format=lambda value: f"{value:.9f}")
    )


if __name__ == "__main__":
    main()
