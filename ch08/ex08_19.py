"""Replicate Exercise 8.19 with constrained least squares.

The code builds the restricted wage model, solves the constrained estimator,
and compares it with unrestricted OLS to make linear restrictions concrete.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


UNRESTRICTED_COLUMNS = [
    "education",
    "experience",
    "experience_sq_100",
    "married1",
    "married2",
    "married3",
    "widowed",
    "divorced",
    "separated",
    "constant",
]


def build_sample() -> tuple[pd.DataFrame, np.ndarray]:
    """Build the unrestricted wage design with separate marital-status effects."""
    df = load_dataset("cps09mar").copy()
    df["wage"] = df["earnings"] / (df["hours"] * df["week"])
    df["lwage"] = np.log(df["wage"])
    df["experience"] = df["age"] - df["education"] - 6
    df["experience_sq_100"] = df["experience"] ** 2 / 100.0

    sample = df["race"].eq(1) & df["hisp"].eq(1) & df["female"].eq(0)
    data = df.loc[sample].dropna(
        subset=["lwage", "education", "experience", "experience_sq_100", "marital"]
    ).copy()

    x = pd.DataFrame(
        {
            "education": data["education"].astype(float),
            "experience": data["experience"].astype(float),
            "experience_sq_100": data["experience_sq_100"].astype(float),
            "married1": data["marital"].eq(1).astype(float),
            "married2": data["marital"].eq(2).astype(float),
            "married3": data["marital"].eq(3).astype(float),
            "widowed": data["marital"].eq(4).astype(float),
            "divorced": data["marital"].eq(5).astype(float),
            "separated": data["marital"].eq(6).astype(float),
            "constant": 1.0,
        }
    )
    y = data["lwage"].to_numpy()
    return x, y


def restriction_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Encode the two equality restrictions R beta = c."""
    r = np.zeros((2, len(UNRESTRICTED_COLUMNS)))
    r[0, UNRESTRICTED_COLUMNS.index("married1")] = 1.0
    r[0, UNRESTRICTED_COLUMNS.index("widowed")] = -1.0
    r[1, UNRESTRICTED_COLUMNS.index("divorced")] = 1.0
    r[1, UNRESTRICTED_COLUMNS.index("separated")] = -1.0
    c = np.zeros(2)
    return r, c


def cls_design(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reparameterize the design so equality constraints hold by construction."""
    z = pd.DataFrame(
        {
            "education": x["education"],
            "experience": x["experience"],
            "experience_sq_100": x["experience_sq_100"],
            "married1_widowed": x["married1"] + x["widowed"],
            "married2": x["married2"],
            "married3": x["married3"],
            "divorced_separated": x["divorced"] + x["separated"],
            "constant": x["constant"],
        }
    )
    t = np.zeros((len(UNRESTRICTED_COLUMNS), z.shape[1]))
    t[UNRESTRICTED_COLUMNS.index("education"), 0] = 1.0
    t[UNRESTRICTED_COLUMNS.index("experience"), 1] = 1.0
    t[UNRESTRICTED_COLUMNS.index("experience_sq_100"), 2] = 1.0
    t[UNRESTRICTED_COLUMNS.index("married1"), 3] = 1.0
    t[UNRESTRICTED_COLUMNS.index("widowed"), 3] = 1.0
    t[UNRESTRICTED_COLUMNS.index("married2"), 4] = 1.0
    t[UNRESTRICTED_COLUMNS.index("married3"), 5] = 1.0
    t[UNRESTRICTED_COLUMNS.index("divorced"), 6] = 1.0
    t[UNRESTRICTED_COLUMNS.index("separated"), 6] = 1.0
    t[UNRESTRICTED_COLUMNS.index("constant"), 7] = 1.0
    return z.to_numpy(), t


def constrained_gamma(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the equality-plus-inequality constrained least-squares problem."""
    cls_fit = sm.OLS(y, z).fit()
    start = np.asarray(cls_fit.params, dtype=float).copy()

    def objective(gamma: np.ndarray) -> float:
        """Return the residual sum of squares for a candidate constrained parameter."""
        residual = y - z @ gamma
        return float(residual @ residual)

    constraints = [
        # Monotonicity restrictions are written in terms of the reparameterized slopes.
        {"type": "ineq", "fun": lambda gamma: gamma[1]},
        {"type": "ineq", "fun": lambda gamma: gamma[1] + gamma[2]},
    ]

    result = minimize(objective, start, method="SLSQP", constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def format_table(beta: np.ndarray, se: np.ndarray) -> pd.DataFrame:
    """Attach coefficient labels to a coefficient/standard-error pair."""
    return pd.DataFrame({"coef": beta, "robust_se_HC1": se}, index=UNRESTRICTED_COLUMNS)


def main() -> None:
    x, y = build_sample()
    fit = sm.OLS(y, x.to_numpy()).fit()
    beta = np.asarray(fit.params, dtype=float)
    cov = np.asarray(fit.get_robustcov_results(cov_type="HC1").cov_params(), dtype=float)
    se = np.sqrt(np.diag(cov))

    r, c = restriction_matrix()

    z, t = cls_design(x)
    fit_cls_reparam = sm.OLS(y, z).fit()
    cov_cls_reparam = np.asarray(
        fit_cls_reparam.get_robustcov_results(cov_type="HC1").cov_params(),
        dtype=float,
    )
    beta_cls = t @ np.asarray(fit_cls_reparam.params, dtype=float)
    cov_cls = t @ cov_cls_reparam @ t.T
    se_cls = np.sqrt(np.diag(cov_cls))

    # EMD updates the unrestricted estimate using the robust covariance metric.
    middle = np.linalg.inv(r @ cov @ r.T)
    beta_emd = beta - cov @ r.T @ middle @ (r @ beta - c)
    cov_emd = cov - cov @ r.T @ middle @ r @ cov
    se_emd = np.sqrt(np.diag(cov_emd))

    gamma_ineq = constrained_gamma(z, y)
    beta_ineq = t @ gamma_ineq

    print("Exercise 8.19")
    print(f"n = {len(y)}")
    print("\nUnrestricted OLS")
    print(format_table(beta, se).to_string(float_format=lambda value: f"{value:.9f}"))
    print("\nCLS with beta4=beta7 and beta8=beta9")
    print(format_table(beta_cls, se_cls).to_string(float_format=lambda value: f"{value:.9f}"))
    print("\nEMD with beta4=beta7 and beta8=beta9")
    print(format_table(beta_emd, se_emd).to_string(float_format=lambda value: f"{value:.9f}"))
    print("\nEquality + inequality constrained estimates")
    print(pd.Series(beta_ineq, index=UNRESTRICTED_COLUMNS).to_string(float_format=lambda value: f"{value:.9f}"))
    print("\nInequality checks")
    print(f"beta2 = {beta_ineq[UNRESTRICTED_COLUMNS.index('experience')]:.9f}")
    print(
        "beta2 + beta3 = "
        f"{beta_ineq[UNRESTRICTED_COLUMNS.index('experience')] + beta_ineq[UNRESTRICTED_COLUMNS.index('experience_sq_100')]:.9f}"
    )


if __name__ == "__main__":
    main()
