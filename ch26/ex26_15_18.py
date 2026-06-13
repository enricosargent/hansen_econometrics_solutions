"""Replicate Exercises 26.15-26.18 with travel-mode choice models.

The code prepares Koppelman alternatives and compares conditional logit,
nested logit, mixed logit, and multinomial probit estimates.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.ch26.common import (
    ChoiceData,
    fit_mixed,
    fit_mnl,
    fit_mnp,
    fit_nested,
    halton_draws,
    load_koppelman_long,
    unpack_mnp_params,
    covariance_to_mnp_moments,
)


CASE_FEATURES = ["income", "urban", "const"]


def koppelman_data(spec: str) -> ChoiceData:
    """Build a travel-mode choice dataset for the requested utility specification."""
    data = load_koppelman_long().copy()
    if spec == "cl_base":
        common_names = ["cost", "intime"]
    elif spec == "cl_outtime":
        common_names = ["cost", "intime", "outtime"]
    elif spec == "cl_total_time":
        data["time"] = data["intime"] + data["outtime"]
        common_names = ["cost", "time"]
    elif spec == "cl_logs":
        # Log specifications reinterpret slopes as semi-elasticities of utility.
        data["log_cost"] = np.log(data["cost"])
        data["log_intime"] = np.log(data["intime"])
        common_names = ["log_cost", "log_intime"]
    elif spec == "mixed_base":
        common_names = ["cost"]
    elif spec == "mixed_total_time":
        data["time"] = data["intime"] + data["outtime"]
        common_names = ["cost"]
    elif spec == "mnp_base":
        common_names = ["cost", "intime"]
    elif spec == "mnp_logs":
        data["log_cost"] = np.log(data["cost"])
        data["log_intime"] = np.log(data["intime"])
        common_names = ["log_cost", "log_intime"]
    else:
        raise ValueError(spec)

    data["const"] = 1.0
    cases = np.sort(data["case"].unique())
    n = len(cases)
    j = 4
    x_common = np.zeros((n, j, len(common_names)), float)
    w_case = np.zeros((n, len(CASE_FEATURES)), float)
    y = np.zeros(n, int)
    by_case = data.sort_values(["case", "alt_index"]).groupby("case", sort=True)
    for idx, (_, group) in enumerate(by_case):
        group = group.sort_values("alt_index")
        x_common[idx] = group[common_names].to_numpy(float)
        w_case[idx] = group.iloc[0][["income", "urban", "const"]].to_numpy(float)
        y[idx] = int(group.loc[group["choice"].eq(1), "alt_index"].iloc[0])
    return ChoiceData(
        y=y,
        x_common=x_common,
        w_case=w_case,
        alt_names=["train", "air", "bus", "car"],
        case_feature_names=CASE_FEATURES.copy(),
        common_feature_names=common_names,
    )


def random_time_matrix(spec: str) -> np.ndarray:
    """Return the alternative-specific time variable used as a random coefficient."""
    data = load_koppelman_long().copy()
    if spec == "base":
        variable = "intime"
        data["timevar"] = data[variable]
    elif spec == "total_time":
        data["timevar"] = data["intime"] + data["outtime"]
    elif spec == "lognormal":
        data["timevar"] = data["intime"]
    else:
        raise ValueError(spec)
    cases = np.sort(data["case"].unique())
    x_random = np.zeros((len(cases), 4), float)
    by_case = data.sort_values(["case", "alt_index"]).groupby("case", sort=True)
    for idx, (_, group) in enumerate(by_case):
        group = group.sort_values("alt_index")
        x_random[idx] = group["timevar"].to_numpy(float)
    return x_random


def conditional_table_start() -> np.ndarray:
    """Starting values close to the conditional-logit table estimates."""
    common = np.array([-0.022, -0.015], float)
    case = np.array(
        [
            0.036,
            0.29,
            -2.15,
            -0.051,
            -0.23,
            -1.79,
            0.008,
            -0.99,
            1.86,
        ],
        float,
    )
    return np.concatenate([common, case])


def nested_table_start() -> np.ndarray:
    """Starting values for the nested-logit likelihood."""
    common = np.array([-0.011, -0.005], float)
    case = np.array(
        [
            0.024,
            0.28,
            -0.46,
            -0.049,
            -0.21,
            -1.55,
            0.017,
            -0.58,
            1.19,
        ],
        float,
    )
    tau = np.array([np.log(0.24 / 0.76)], float)
    return np.concatenate([common, case, tau])


def mixed_table_start() -> np.ndarray:
    """Starting values for the mixed-logit random-time specification."""
    case = np.array(
        [
            0.040,
            0.35,
            -2.72,
            -0.050,
            -0.24,
            -1.82,
            0.008,
            -1.01,
            1.89,
        ],
        float,
    )
    return np.concatenate([np.array([-0.023, -0.014, np.log(0.0048)], float), case])


def cholesky_params(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix into the unconstrained MNP Cholesky parameters."""
    lower = np.linalg.cholesky(covariance)
    return np.array(
        [
            lower[1, 0],
            np.log(lower[1, 1]),
            lower[2, 0],
            lower[2, 1],
            np.log(lower[2, 2]),
        ],
        float,
    )


def mnp_table_start() -> np.ndarray:
    """Starting values for multinomial probit utility and covariance parameters."""
    common = np.array([-0.005, -0.005], float)
    case = np.array(
        [
            0.018,
            -0.38,
            0.32,
            -0.008,
            -0.14,
            -0.23,
            0.013,
            -0.79,
            1.51,
        ],
        float,
    )
    covariance = np.array(
        [
            [2.0, 0.60 * np.sqrt(2.0 * 0.41), 0.99 * np.sqrt(2.0 * 3.8)],
            [0.60 * np.sqrt(2.0 * 0.41), 0.41, 0.60 * np.sqrt(0.41 * 3.8)],
            [0.99 * np.sqrt(2.0 * 3.8), 0.60 * np.sqrt(0.41 * 3.8), 3.8],
        ]
    )
    return np.concatenate([common, case, cholesky_params(covariance)])


def summarize_common(
    fit: dict[str, object],
    feature_names: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Extract common-slope estimates and standard errors for compact tables."""
    out: dict[str, float] = {"loglike": float(fit["loglike"])}
    params = fit["params"]
    ses = fit["se"]
    for index, feature_name in enumerate(feature_names[: len(labels)]):
        label = labels[index]
        out[f"{label}_coef"] = float(params[index])
        out[f"{label}_se"] = float(ses[index])
    return out


def summarize_mixed(
    fit: dict[str, object],
    distribution: str,
) -> dict[str, float]:
    """Summarize the mixed-logit random coefficient distribution."""
    params = fit["params"]
    ses = fit["se"]
    sigma = float(np.exp(params[2]))
    sigma_se = float(np.exp(params[2]) * ses[2])
    summary = {
        "cost_coef": float(params[0]),
        "cost_se": float(ses[0]),
        "time_location": float(params[1]),
        "time_location_se": float(ses[1]),
        "sigma": sigma,
        "sigma_se": sigma_se,
        "loglike": float(fit["loglike"]),
    }
    if distribution == "neg_lognormal":
        summary["time_mean_implied"] = float(-np.exp(params[1] + 0.5 * sigma**2))
    return summary


def summarize_mnp(fit: dict[str, object], n_common: int) -> dict[str, float]:
    """Summarize MNP slopes and covariance moments."""
    base_params, sigma = unpack_mnp_params(fit["params"], koppelman_data("mnp_base"))
    moments = covariance_to_mnp_moments(sigma)
    summary = {
        "cost_coef": float(base_params[0]),
        "cost_se": float(fit["se"][0]),
        "time_coef": float(base_params[1]),
        "time_se": float(fit["se"][1]),
        "loglike": float(fit["loglike"]),
    }
    summary.update(moments)
    return summary


def main() -> None:
    conditional_results = []

    # Exercise 26.15 varies the time and cost specification in conditional logit.
    cl_a_data = koppelman_data("cl_base")
    cl_a_fit = fit_mnl(cl_a_data, start=conditional_table_start())
    conditional_results.append(
        {
            "case": "15(a)",
            **summarize_common(cl_a_fit, cl_a_data.common_feature_names, ["cost", "intime"]),
        }
    )

    cl_b_data = koppelman_data("cl_outtime")
    cl_b_start = np.concatenate([cl_a_fit["params"][:2], np.array([0.0]), cl_a_fit["params"][2:]])
    cl_b_fit = fit_mnl(cl_b_data, start=cl_b_start)
    conditional_results.append(
        {
            "case": "15(b)",
            **summarize_common(cl_b_fit, cl_b_data.common_feature_names, ["cost", "intime", "outtime"]),
        }
    )

    cl_c_data = koppelman_data("cl_total_time")
    cl_c_start = np.concatenate([cl_a_fit["params"][:1], cl_a_fit["params"][1:2], cl_a_fit["params"][2:]])
    cl_c_fit = fit_mnl(cl_c_data, start=cl_c_start)
    conditional_results.append(
        {
            "case": "15(c)",
            **summarize_common(cl_c_fit, cl_c_data.common_feature_names, ["cost", "time"]),
        }
    )

    cl_d_data = koppelman_data("cl_logs")
    cl_d_start = np.concatenate([np.array([-1.0, -1.0]), cl_a_fit["params"][2:]])
    cl_d_fit = fit_mnl(cl_d_data, start=cl_d_start)
    conditional_results.append(
        {
            "case": "15(d)",
            **summarize_common(cl_d_fit, cl_d_data.common_feature_names, ["log_cost", "log_intime"]),
        }
    )

    nested_results = []
    groups_ca_tb = [(1, 3), (0, 2)]
    # The first nest groups car/air and train/bus following the exercise table.
    nl_a_data = cl_a_data
    nl_a_fit = fit_nested(
        nl_a_data,
        groups_ca_tb,
        fixed_taus={1: 1.0},
        start_base=nested_table_start()[:-1],
        start_tau=[0.24],
    )
    nested_results.append(
        {
            "case": "16(a)",
            **summarize_common(nl_a_fit, nl_a_data.common_feature_names, ["cost", "intime"]),
            "taus": ", ".join(f"{tau:.6f}" for tau in nl_a_fit["taus"]),
        }
    )

    nl_b_data = cl_d_data
    nl_b_fit = fit_nested(
        nl_b_data,
        groups_ca_tb,
        fixed_taus={1: 1.0},
        start_base=np.concatenate([np.array([-0.5, -0.5]), nl_a_fit["params"][2:11]]),
        start_tau=[nl_a_fit["taus"][0]],
    )
    nested_results.append(
        {
            "case": "16(b)",
            **summarize_common(nl_b_fit, nl_b_data.common_feature_names, ["log_cost", "log_intime"]),
            "taus": ", ".join(f"{tau:.6f}" for tau in nl_b_fit["taus"]),
        }
    )

    nl_c_fit = fit_nested(
        nl_a_data,
        [(3,), (0, 1, 2)],
        start_base=cl_a_fit["params"],
        start_tau=[0.6],
    )
    nested_results.append(
        {
            "case": "16(c)",
            **summarize_common(nl_c_fit, nl_a_data.common_feature_names, ["cost", "intime"]),
            "taus": ", ".join(f"{tau:.6f}" for tau in nl_c_fit["taus"]),
        }
    )

    nl_d_fit = fit_nested(
        nl_a_data,
        [(1,), (0, 2, 3)],
        start_base=cl_a_fit["params"],
        start_tau=[0.6],
    )
    nested_results.append(
        {
            "case": "16(d)",
            **summarize_common(nl_d_fit, nl_a_data.common_feature_names, ["cost", "intime"]),
            "taus": ", ".join(f"{tau:.6f}" for tau in nl_d_fit["taus"]),
        }
    )

    mixed_results = []
    mx_a_data = koppelman_data("mixed_base")
    mx_a_time = random_time_matrix("base")
    # Gauss-Hermite quadrature integrates over the random time coefficient.
    mx_a_fit = fit_mixed(
        mx_a_data,
        mx_a_time,
        p_fixed=1,
        n_nodes=30,
        distribution="normal",
        start=mixed_table_start(),
    )
    mixed_results.append({"case": "17(a)", **summarize_mixed(mx_a_fit, "normal")})

    mx_b_time = random_time_matrix("total_time")
    mx_b_fit = fit_mixed(
        mx_a_data,
        mx_b_time,
        p_fixed=1,
        n_nodes=30,
        distribution="normal",
        start=mx_a_fit["params"],
    )
    mixed_results.append({"case": "17(b)", **summarize_mixed(mx_b_fit, "normal")})

    mx_c_time = random_time_matrix("lognormal")
    mx_c_fit = fit_mixed(
        mx_a_data,
        mx_c_time,
        p_fixed=1,
        n_nodes=30,
        distribution="neg_lognormal",
        start=np.concatenate([np.array([mx_a_fit["params"][0], -4.0, np.log(0.2)]), mx_a_fit["params"][3:]]),
    )
    mixed_results.append({"case": "17(c)", **summarize_mixed(mx_c_fit, "neg_lognormal")})

    mnp_results = []
    # Fixed Halton draws make the simulated MNP likelihood reproducible.
    mnp_draws = halton_draws(cl_a_data.n, 120, 3, seed=26018)
    mp_a_data = koppelman_data("mnp_base")
    mp_a_fit = fit_mnp(mp_a_data, mnp_draws, start=mnp_table_start())
    mnp_results.append({"case": "18(a)", **summarize_mnp(mp_a_fit, 2)})

    mp_b_data = koppelman_data("mnp_logs")
    mp_b_start = np.concatenate([np.array([-0.5, -0.5]), mp_a_fit["params"][2:]])
    mp_b_fit = fit_mnp(mp_b_data, mnp_draws, start=mp_b_start)
    mnp_results.append({"case": "18(b)", **summarize_mnp(mp_b_fit, 2)})

    print("Exercise 26.15")
    print(pd.DataFrame(conditional_results).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Exercise 26.16")
    print(pd.DataFrame(nested_results).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Exercise 26.17")
    print(pd.DataFrame(mixed_results).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Exercise 26.18")
    print(pd.DataFrame(mnp_results).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
