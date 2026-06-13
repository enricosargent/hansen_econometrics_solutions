"""Replicate Chapter 18 panel-data regression exercises.

The code estimates fixed-effects style regressions and Wald tests, emphasizing
how within variation and clustered uncertainty enter empirical panel work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


def fit_ols(
    frame: pd.DataFrame,
    outcome_name: str,
    regressor_names: list[str],
    fixed_effects: list[str] | None = None,
    cluster_name: str | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS with optional fixed effects and either HC1 or clustered SEs."""
    selected_names = list(dict.fromkeys([outcome_name] + regressor_names + (fixed_effects or []) + ([cluster_name] if cluster_name else [])))
    work = frame[selected_names].dropna().copy()
    design_parts = [work[regressor_names].astype(float)]
    for fixed_effect_name in fixed_effects or []:
        # Dummy fixed effects make the within comparison explicit in the design matrix.
        dummies = pd.get_dummies(work[fixed_effect_name], prefix=fixed_effect_name, drop_first=True, dtype=float)
        design_parts.append(dummies)
    design = pd.concat(design_parts, axis=1)
    design = sm.add_constant(design, has_constant="add")
    outcome = work[outcome_name].astype(float)
    model = sm.OLS(outcome, design)
    if cluster_name:
        return model.fit(cov_type="cluster", cov_kwds={"groups": work[cluster_name]})
    return model.fit(cov_type="HC1")


def wald_pvalue(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    names: list[str],
) -> tuple[float, float]:
    """Compute a chi-square Wald test for a block of named coefficients."""
    parameter_names = list(result.params.index)
    restriction = np.zeros((len(names), len(parameter_names)))
    for row_index, name in enumerate(names):
        restriction[row_index, parameter_names.index(name)] = 1.0
    covariance = result.cov_params().to_numpy()
    estimate = result.params.to_numpy()
    diff = restriction @ estimate
    variance = restriction @ covariance @ restriction.T
    statistic = float(diff.T @ np.linalg.pinv(variance) @ diff)
    pvalue = float(stats.chi2.sf(statistic, len(names)))
    return statistic, pvalue


def print_coefficient(
    label: str,
    result: sm.regression.linear_model.RegressionResultsWrapper,
    name: str,
) -> None:
    """Print one coefficient with its standard error and p-value."""
    print(f"{label}: {name} = {result.params[name]:.6f}, se = {result.bse[name]:.6f}, p = {result.pvalues[name]:.6f}")


def exercise_18_6() -> None:
    """Reproduce the Card-Krueger difference-in-differences price exercise."""
    data = load_dataset("CK1994").copy()
    data["price"] = data["priceentree"] + data["pricefry"] + data["pricesoda"]
    data = data[np.isfinite(data["price"])]
    balanced_stores = data.groupby("store")["time"].nunique()
    balanced_ids = balanced_stores[balanced_stores == 2].index
    data = data[data["store"].isin(balanced_ids)].copy()
    data["post"] = data["time"]
    data["treated"] = data["state"]
    data["treatment"] = data["treated"] * data["post"]
    print(f"Exercise 18.6: balanced CK1994 price sample, n={len(data)}, stores={data['store'].nunique()}")

    # The two-by-two table shows the DID estimand before regression adjustment.
    table = data.groupby(["treated", "post"])["price"].mean().unstack()
    pa_before = table.loc[0.0, 0.0]
    pa_after = table.loc[0.0, 1.0]
    nj_before = table.loc[1.0, 0.0]
    nj_after = table.loc[1.0, 1.0]
    nj_change = nj_after - nj_before
    pa_change = pa_after - pa_before
    did = nj_change - pa_change
    print("Table 18.1 analog for meal prices")
    print(f"  NJ before={nj_before:.6f}, after={nj_after:.6f}, difference={nj_change:.6f}")
    print(f"  PA before={pa_before:.6f}, after={pa_after:.6f}, difference={pa_change:.6f}")
    print(f"  difference-in-differences={did:.6f}")

    result_basic = fit_ols(data, "price", ["treated", "post", "treatment"], cluster_name="store")
    result_state_fe = fit_ols(data, "price", ["post", "treatment"], fixed_effects=["state"], cluster_name="store")
    result_store_fe = fit_ols(data, "price", ["post", "treatment"], fixed_effects=["store"], cluster_name="store")
    print_coefficient("Regression 18.2 analog", result_basic, "treatment")
    print_coefficient("State fixed-effect analog", result_state_fe, "treatment")
    print_coefficient("Restaurant fixed-effect analog", result_store_fe, "treatment")

    regions = ["southj", "centralj", "northj", "pa1", "pa2"]
    print("Table 18.2 analog by region")
    for region_name in regions:
        region_sample = data[data[region_name] == 1]
        before = float(region_sample.loc[region_sample["post"] == 0, "price"].mean())
        after = float(region_sample.loc[region_sample["post"] == 1, "price"].mean())
        print(f"  {region_name:8s} before={before:.6f}, after={after:.6f}, difference={after-before:.6f}")

    data["centralj_post"] = data["centralj"] * data["post"]
    data["northj_post"] = data["northj"] * data["post"]
    treatment_test = fit_ols(
        data,
        "price",
        ["post", "treatment", "centralj_post", "northj_post"],
        fixed_effects=["store"],
        cluster_name="store",
    )
    statistic, pvalue = wald_pvalue(treatment_test, ["centralj_post", "northj_post"])
    print(f"Homogeneous treatment effects: chi2={statistic:.6f}, p={pvalue:.6f}")

    data["pa2_post"] = data["pa2"] * data["post"]
    control_test = fit_ols(
        data,
        "price",
        ["post", "treatment", "pa2_post"],
        fixed_effects=["store"],
        cluster_name="store",
    )
    print_coefficient("Equal control effects", control_test, "pa2_post")


def exercise_18_7() -> None:
    """Estimate treatment effects by distance in the Draca-Skouras panel."""
    data = load_dataset("DS2004").copy()
    data = data[data["month"] != 7].copy()
    data["post"] = (data["month"] >= 8).astype(float)
    data["same_treatment"] = data["sameblock"] * data["post"]
    data["one_treatment"] = data["oneblock"] * data["post"]
    print(f"Exercise 18.7: DS2004 without July, n={len(data)}, blocks={data['block'].nunique()}")

    # The hand-computed DID clarifies what the fixed-effect regression generalizes.
    distance_sample = data[data["sameblock"] == 0].copy()
    table = distance_sample.groupby(["oneblock", "post"])["thefts"].mean().unstack()
    far_before = table.loc[0.0, 0.0]
    far_after = table.loc[0.0, 1.0]
    one_before = table.loc[1.0, 0.0]
    one_after = table.loc[1.0, 1.0]
    one_change = one_after - one_before
    far_change = far_after - far_before
    did = one_change - far_change
    print("One-block table")
    print(f"  one block before={one_before:.6f}, after={one_after:.6f}, difference={one_change:.6f}")
    print(f"  farther before={far_before:.6f}, after={far_after:.6f}, difference={far_change:.6f}")
    print(f"  difference-in-differences={did:.6f}")

    result = fit_ols(
        data,
        "thefts",
        ["same_treatment", "one_treatment"],
        fixed_effects=["block", "month"],
        cluster_name="block",
    )
    print_coefficient("Block/month fixed effects", result, "same_treatment")
    print_coefficient("Block/month fixed effects", result, "one_treatment")


def exercise_18_8() -> None:
    """Estimate beer-demand policy regressions with state and year controls."""
    data = load_dataset("BMN2016").copy()
    regressors = ["beeronsun", "beeroffsun", "unempw", "beerOnOutflows", "beerOffOutflows"]
    base = fit_ols(data, "logbeer", regressors, fixed_effects=["id", "year"], cluster_name="id")
    print("Exercise 18.8: BMN2016 beer sales, state and year fixed effects")
    for name in regressors:
        print_coefficient("Base model", base, name)

    data["trend"] = data["year"] - data["year"].min()
    trend_parts = []
    # State-specific trends allow each state to have its own smooth pretrend.
    for state_id in sorted(data["id"].dropna().unique()):
        name = f"trend_id_{int(state_id)}"
        data[name] = (data["id"] == state_id).astype(float) * data["trend"]
        trend_parts.append(name)
    trend = fit_ols(data, "logbeer", regressors + trend_parts, fixed_effects=["id", "year"], cluster_name="id")
    print("Exercise 18.8: BMN2016 beer sales with state-specific linear trends")
    for name in regressors:
        print_coefficient("Trend model", trend, name)


def main() -> None:
    exercise_18_6()
    print()
    exercise_18_7()
    print()
    exercise_18_8()


if __name__ == "__main__":
    main()
