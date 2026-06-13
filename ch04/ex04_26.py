"""Replicate Exercise 4.26 with standardized CPS wage regressors.

The code rescales key covariates, fits the wage model, and prints coefficients
whose magnitudes can be compared on a common standard-deviation scale.
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


def standardized(series: pd.Series) -> pd.Series:
    """Put a variable on a mean-zero, unit-standard-deviation scale."""
    return (series - series.mean()) / series.std(ddof=1)


def main() -> None:
    df = load_dataset("DDK2011").copy()
    df["testscore"] = standardized(df["totalscore"])

    # Start with the simple tracking-only equation before adding controls.
    base = df.dropna(subset=["testscore", "tracking", "schoolid"]).copy()
    base_x = np.column_stack([np.ones(len(base)), base["tracking"].astype(float).to_numpy()])
    base_y = base["testscore"].to_numpy()
    base_fit = sm.OLS(base_y, base_x).fit()
    base_hc1 = base_fit.get_robustcov_results(cov_type="HC1")
    base_cluster = base_fit.get_robustcov_results(
        cov_type="cluster",
        groups=base["schoolid"].to_numpy(),
        use_correction=True,
        df_correction=True,
    )

    keep = ["testscore", "tracking", "agetest", "girl", "etpteacher", "percentile", "schoolid"]
    data = df.dropna(subset=keep).copy()
    # Controls adjust for baseline covariates while clustering respects schools.
    x = pd.DataFrame(
        {
            "tracking": data["tracking"].astype(float),
            "age": data["agetest"].astype(float),
            "girl": data["girl"].astype(float),
            "contract_teacher": data["etpteacher"].astype(float),
            "percentile": data["percentile"].astype(float),
            "constant": 1.0,
        }
    )
    y = data["testscore"].to_numpy()
    fit = sm.OLS(y, x.to_numpy()).fit()
    robust = fit.get_robustcov_results(cov_type="HC1").bse
    clustered = fit.get_robustcov_results(
        cov_type="cluster",
        groups=data["schoolid"].to_numpy(),
        use_correction=True,
        df_correction=True,
    ).bse

    results = pd.DataFrame(
        {
            "coef": fit.params,
            "robust_HC1": robust,
            "cluster_school": clustered,
        },
        index=x.columns,
    )
    results["absolute_change"] = results["cluster_school"] - results["robust_HC1"]
    results["ratio"] = results["cluster_school"] / results["robust_HC1"]

    reduced_x = np.column_stack([np.ones(len(data)), data["tracking"].astype(float).to_numpy()])
    reduced_fit = sm.OLS(y, reduced_x).fit()

    print("Exercise 4.26")
    print(f"controlled-sample n = {len(data)}")
    print(f"number of schools = {data['schoolid'].nunique()}")
    print(results.to_string(float_format=lambda value: f"{value:.9f}"))
    print("\nBaseline equation (4.57) reproduction on full available sample")
    print(
        f"coef = [{base_fit.params[0]:.9f}, {base_fit.params[1]:.9f}], "
        f"HC1 = [{base_hc1.bse[0]:.9f}, {base_hc1.bse[1]:.9f}], "
        f"cluster = [{base_cluster.bse[0]:.9f}, {base_cluster.bse[1]:.9f}]"
    )
    print("Reduced-sample tracking-only regression")
    print(
        f"coef = [{reduced_fit.params[0]:.9f}, {reduced_fit.params[1]:.9f}], "
        f"HC1 = [{reduced_fit.get_robustcov_results(cov_type='HC1').bse[0]:.9f}, "
        f"{reduced_fit.get_robustcov_results(cov_type='HC1').bse[1]:.9f}]"
    )


if __name__ == "__main__":
    main()
