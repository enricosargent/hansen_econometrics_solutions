"""Replicate Exercise 9.26 with joint tests in a wage regression.

The script estimates the model, forms the relevant restrictions, and prints the
test statistics used to interpret groups of regressors together.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


UNRESTRICTED_COLUMNS = ["const", "logQ", "logPL", "logPK", "logPF"]


def main() -> None:
    df = load_dataset("Nerlove1963").copy()
    data = df.dropna(subset=["cost", "output", "Plabor", "Pcapital", "Pfuel"]).copy()

    # Logs turn the cost equation into the Cobb-Douglas linear regression.
    data["logC"] = np.log(data["cost"].astype(float))
    data["logQ"] = np.log(data["output"].astype(float))
    data["logPL"] = np.log(data["Plabor"].astype(float))
    data["logPK"] = np.log(data["Pcapital"].astype(float))
    data["logPF"] = np.log(data["Pfuel"].astype(float))

    y = data["logC"].astype(float)
    x_unrestricted = pd.DataFrame(
        {
            "const": 1.0,
            "logQ": data["logQ"],
            "logPL": data["logPL"],
            "logPK": data["logPK"],
            "logPF": data["logPF"],
        }
    )
    unrestricted_fit = sm.OLS(y, x_unrestricted).fit(cov_type="HC2")

    y_constrained = data["logC"] - data["logPF"]
    x_constrained = pd.DataFrame(
        {
            "const": 1.0,
            "logQ": data["logQ"],
            "logPL_minus_logPF": data["logPL"] - data["logPF"],
            "logPK_minus_logPF": data["logPK"] - data["logPF"],
        }
    )
    constrained_fit = sm.OLS(y_constrained, x_constrained).fit(cov_type="HC2")

    # The transform maps constrained coefficients back to the unrestricted labels.
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, -1.0],
        ]
    )
    offset = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    beta_cls = offset + transform @ constrained_fit.params.to_numpy()
    cov_cls = transform @ constrained_fit.cov_params().to_numpy() @ transform.T
    se_cls = np.sqrt(np.diag(cov_cls))

    restriction = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]])
    target = np.array([1.0])
    cov_unrestricted = unrestricted_fit.cov_params().to_numpy()
    beta_unrestricted = unrestricted_fit.params.to_numpy()
    # EMD imposes the restriction using the robust covariance as the distance metric.
    middle = np.linalg.inv(restriction @ cov_unrestricted @ restriction.T)
    beta_emd = beta_unrestricted - cov_unrestricted @ restriction.T @ middle @ (
        restriction @ beta_unrestricted - target
    )
    cov_emd = cov_unrestricted - cov_unrestricted @ restriction.T @ middle @ restriction @ cov_unrestricted
    se_emd = np.sqrt(np.diag(cov_emd))

    wald = unrestricted_fit.wald_test((restriction, target), scalar=True)
    md_statistic = float((restriction @ beta_unrestricted - target).T @ middle @ (restriction @ beta_unrestricted - target))

    unrestricted_table = pd.DataFrame(
        {"coef": unrestricted_fit.params, "se_hc2": unrestricted_fit.bse}
    )
    cls_table = pd.DataFrame(
        {"coef": beta_cls, "se_hc2": se_cls},
        index=UNRESTRICTED_COLUMNS,
    )
    emd_table = pd.DataFrame(
        {"coef": beta_emd, "se_hc2": se_emd},
        index=UNRESTRICTED_COLUMNS,
    )

    print("Exercise 9.26")
    print(f"n = {len(data)}")
    print()
    print("Unrestricted Cobb-Douglass regression")
    print(unrestricted_table.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print("Constrained least squares under logPL + logPK + logPF coefficient sum = 1")
    print(cls_table.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print("Efficient minimum distance under the same restriction")
    print(emd_table.to_string(float_format=lambda value: f"{value:.9f}"))
    print()
    print(f"Wald statistic = {wald.statistic:.9f}, p-value = {wald.pvalue:.9f}")
    print(f"Minimum-distance statistic = {md_statistic:.9f}")


if __name__ == "__main__":
    main()
