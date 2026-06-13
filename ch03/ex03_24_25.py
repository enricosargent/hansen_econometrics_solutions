"""Replicate Exercises 3.24-3.25 with CPS wages and OLS geometry.

The script builds Hansen's wage sample, estimates the log-wage regression,
and prints residual identities that teach the projection interpretation of OLS.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


def main() -> None:
    # Build the same log-wage variables and sample restrictions used in the text.
    df = load_dataset("cps09mar")
    df = df.copy()
    df["wage"] = df["earnings"] / (df["hours"] * df["week"])
    df["lwage"] = np.log(df["wage"])
    df["experience"] = df["age"] - df["education"] - 6
    df["exp2"] = df["experience"] ** 2 / 100

    sample = (
        df["race"].eq(4)
        & df["marital"].eq(7)
        & df["female"].eq(0)
        & (df["experience"] < 45)
    )
    data = df.loc[sample].dropna(subset=["lwage", "education", "experience", "exp2"])

    # Estimate the wage equation with the constant placed last to match the manual.
    y = data["lwage"].to_numpy()
    x = np.column_stack(
        [
            data["education"],
            data["experience"],
            data["exp2"],
            np.ones(len(data)),
        ]
    )
    fit = sm.OLS(y, x).fit()

    # The residual-on-residual regression is the Frisch-Waugh-Lovell check.
    z = np.column_stack([data["experience"], data["exp2"], np.ones(len(data))])
    y_residual = sm.OLS(y, z).fit().resid
    education_residual = sm.OLS(data["education"].to_numpy(), z).fit().resid
    residual_regression = sm.OLS(y_residual, education_residual[:, None]).fit()

    print("Exercise 3.24")
    print(f"n = {len(data)}")
    print("coefficients [education, experience, experience^2/100, constant]")
    print(np.array2string(fit.params, precision=9))
    print(f"R2 = {fit.rsquared:.9f}")
    print(f"SSE = {fit.ssr:.9f}")
    print(f"residual-regression education slope = {residual_regression.params[0]:.9f}")
    print(f"residual-regression R2 = {residual_regression.rsquared:.9f}")
    print(f"residual-regression SSE = {residual_regression.ssr:.9f}")

    residual = fit.resid
    fitted = fit.fittedvalues
    education = data["education"].to_numpy()
    experience = data["experience"].to_numpy()

    checks = {
        # OLS residuals are orthogonal to included regressors and fitted values.
        "sum e_i": residual.sum(),
        "sum education_i e_i": (education * residual).sum(),
        "sum experience_i e_i": (experience * residual).sum(),
        "sum education_i^2 e_i": ((education**2) * residual).sum(),
        "sum experience_i^2 e_i": ((experience**2) * residual).sum(),
        "sum fitted_i e_i": (fitted * residual).sum(),
        "sum e_i^2": (residual**2).sum(),
    }

    print("\nExercise 3.25")
    for label, value in checks.items():
        print(f"{label}: {value:.12g}")


if __name__ == "__main__":
    main()
