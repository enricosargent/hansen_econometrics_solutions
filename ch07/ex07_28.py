"""Replicate Exercise 7.28 on robust inference in a wage regression.

The script estimates the CPS model and contrasts conventional and robust
standard errors so the printed output highlights heteroskedasticity's role.
"""

from __future__ import annotations

from pathlib import Path
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


def critical(probability: float) -> float:
    """Return a standard-normal critical value for confidence intervals."""
    return NormalDist().inv_cdf(probability)


def main() -> None:
    # Build the log-wage regression variables before forming nonlinear functions.
    df = load_dataset("cps09mar").copy()
    df["wage"] = df["earnings"] / (df["hours"] * df["week"])
    df["lwage"] = np.log(df["wage"])
    df["experience"] = df["age"] - df["education"] - 6
    df["experience_sq_100"] = df["experience"] ** 2 / 100.0

    sample = df["race"].eq(1) & df["hisp"].eq(1) & df["female"].eq(0)
    data = df.loc[sample].dropna(subset=["lwage", "education", "experience", "experience_sq_100"]).copy()

    x = pd.DataFrame(
        {
            "education": data["education"].astype(float),
            "experience": data["experience"].astype(float),
            "experience_sq_100": data["experience_sq_100"].astype(float),
            "constant": 1.0,
        }
    )
    y = data["lwage"].to_numpy()
    fit = sm.OLS(y, x.to_numpy()).fit()
    cov_hc1 = fit.get_robustcov_results(cov_type="HC1").cov_params()
    cov_homo = fit.cov_params()
    se_hc1 = np.sqrt(np.diag(cov_hc1))

    beta = fit.params
    b1, b2, b3, b4 = beta

    x_exp = 10.0
    denom = b2 + (2.0 * x_exp / 100.0) * b3
    theta = b1 / denom
    # The gradient maps robust covariance of beta into uncertainty for theta.
    grad_theta = np.array(
        [
            1.0 / denom,
            -b1 / denom**2,
            -(2.0 * x_exp / 100.0) * b1 / denom**2,
            0.0,
        ]
    )
    theta_se = float(np.sqrt(grad_theta @ cov_hc1 @ grad_theta))
    z90 = critical(0.95)
    theta_ci = (theta - z90 * theta_se, theta + z90 * theta_se)

    x_mean = np.array([12.0, 20.0, 20.0**2 / 100.0, 1.0])
    m_hat = float(x_mean @ beta)
    m_se = float(np.sqrt(x_mean @ cov_hc1 @ x_mean))
    z95 = critical(0.975)
    m_ci = (m_hat - z95 * m_se, m_hat + z95 * m_se)

    x_forecast = np.array([16.0, 5.0, 5.0**2 / 100.0, 1.0])
    f_hat = float(x_forecast @ beta)
    # Forecast intervals add irreducible residual variance to coefficient uncertainty.
    sigma2_hat = float(fit.resid @ fit.resid) / (len(data) - x.shape[1])
    forecast_var = sigma2_hat + float(x_forecast @ cov_homo @ x_forecast)
    forecast_se = np.sqrt(forecast_var)
    z80 = critical(0.90)
    logwage_interval = (f_hat - z80 * forecast_se, f_hat + z80 * forecast_se)
    wage_interval = (np.exp(logwage_interval[0]), np.exp(logwage_interval[1]))

    results = pd.DataFrame(
        {
            "coef": beta,
            "robust_se_HC1": se_hc1,
        },
        index=x.columns,
    )

    print("Exercise 7.28")
    print(f"n = {len(data)}")
    print(results.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print(f"theta_hat = {theta:.9f}")
    print(
        "grad_theta = ["
        + ", ".join(f"{value:.9f}" for value in grad_theta)
        + "]"
    )
    print(f"se(theta_hat) = {theta_se:.9f}")
    print(f"90% CI for theta = ({theta_ci[0]:.9f}, {theta_ci[1]:.9f})")
    print()
    print(f"m_hat(education=12, experience=20) = {m_hat:.9f}")
    print(f"se(m_hat) = {m_se:.9f}")
    print(f"95% CI for m_hat = ({m_ci[0]:.9f}, {m_ci[1]:.9f})")
    print()
    print(f"forecast mean log wage = {f_hat:.9f}")
    print(f"sigma2_hat = {sigma2_hat:.9f}")
    print(f"forecast_se = {forecast_se:.9f}")
    print(
        f"80% forecast interval for log wage = ({logwage_interval[0]:.9f}, {logwage_interval[1]:.9f})"
    )
    print(
        f"80% forecast interval for wage = ({wage_interval[0]:.9f}, {wage_interval[1]:.9f})"
    )


if __name__ == "__main__":
    main()
