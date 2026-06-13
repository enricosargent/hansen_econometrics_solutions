"""Replicate Exercises 26.12-26.14 with multinomial and nested logit models.

The script models marital status choices, computes age profiles and marginal
effects, and compares candidate nests by likelihood.
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
    MARITAL_LABELS,
    add_intercept,
    fit_mnl,
    fit_nested,
    load_cps_marital,
    mnl_probabilities,
    quadratic_spline,
)


def cps_choice_data(frame: pd.DataFrame, regressors: np.ndarray, feature_names: list[str]) -> ChoiceData:
    """Wrap CPS marital outcomes and case-specific regressors as ChoiceData."""
    n = len(frame)
    return ChoiceData(
        y=frame["marital4"].to_numpy(int),
        x_common=np.zeros((n, len(MARITAL_LABELS), 0), float),
        w_case=regressors.astype(float),
        alt_names=MARITAL_LABELS.copy(),
        case_feature_names=feature_names,
        common_feature_names=[],
    )


def age_grid_probabilities(fit: dict[str, object], ages: list[int], knots: tuple[float, ...]) -> pd.DataFrame:
    """Predict fitted marital-status probabilities at selected ages."""
    grid = np.array(ages, float)
    regressors = add_intercept(quadratic_spline(grid, knots))
    data = ChoiceData(
        y=np.zeros(len(grid), int),
        x_common=np.zeros((len(grid), len(MARITAL_LABELS), 0), float),
        w_case=regressors,
        alt_names=MARITAL_LABELS.copy(),
        case_feature_names=["const", "age", "age_sq", "age40_sq"],
        common_feature_names=[],
    )
    probs = mnl_probabilities(fit["params"], data)
    rows = []
    for idx, age in enumerate(ages):
        row = {"age": age}
        for alt_index, alt_name in enumerate(MARITAL_LABELS):
            row[alt_name] = probs[idx, alt_index]
        rows.append(row)
    return pd.DataFrame(rows)


def mnl_case_coefficients(params: np.ndarray, q_case: int, j_alt: int) -> np.ndarray:
    """Restore the normalized zero coefficient vector for the base alternative."""
    beta = np.zeros((j_alt, q_case), float)
    beta[1:] = params.reshape(j_alt - 1, q_case)
    return beta


def average_marginal_effects(data: ChoiceData, fit: dict[str, object]) -> pd.DataFrame:
    """Average MNL derivatives of each choice probability with respect to covariates."""
    probabilities = mnl_probabilities(fit["params"], data)
    q_case = data.q_case
    beta = mnl_case_coefficients(fit["params"], q_case, data.j)
    averages = []
    weighted_beta = probabilities @ beta
    for alt in range(data.j):
        delta = probabilities[:, [alt]] * (beta[alt] - weighted_beta)
        averages.append(delta.mean(axis=0))
    table = pd.DataFrame(averages, columns=data.case_feature_names, index=data.alt_names)
    return table


def coefficient_table(data: ChoiceData, fit: dict[str, object]) -> pd.DataFrame:
    """Format alternative-specific MNL coefficients and standard errors."""
    q_case = data.q_case
    params = fit["params"]
    ses = fit["se"]
    rows = []
    for alt in range(1, data.j):
        for feature_index, feature_name in enumerate(data.case_feature_names):
            position = (alt - 1) * q_case + feature_index
            rows.append(
                {
                    "alternative": data.alt_names[alt],
                    "feature": feature_name,
                    "coef": params[position],
                    "se": ses[position],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    cps = load_cps_marital()

    knots = (40.0,)
    age_feature_names = ["const", "age", "age_sq", "age40_sq"]

    men = cps.loc[cps["female"].eq(0.0)].copy()
    women = cps.loc[cps["female"].eq(1.0)].copy()

    # The age spline lets choice probabilities bend after age 40.
    men_data = cps_choice_data(men, add_intercept(quadratic_spline(men["age"].to_numpy(float), knots)), age_feature_names)
    women_data = cps_choice_data(
        women,
        add_intercept(quadratic_spline(women["age"].to_numpy(float), knots)),
        age_feature_names,
    )

    men_fit = fit_mnl(men_data)
    women_fit = fit_mnl(women_data)

    ages = [25, 35, 45, 55]
    men_grid = age_grid_probabilities(men_fit, ages, knots)
    women_grid = age_grid_probabilities(women_fit, ages, knots)

    women35 = women.loc[women["age"].le(35.0)].copy()
    women35_features = add_intercept(women35[["age", "education"]].to_numpy(float))
    women35_data = cps_choice_data(women35, women35_features, ["const", "age", "education"])
    women35_fit = fit_mnl(women35_data)
    women35_coef = coefficient_table(women35_data, women35_fit)
    women35_ame = average_marginal_effects(women35_data, women35_fit).reset_index().rename(
        columns={"index": "alternative"}
    )

    candidate_groups = [
        ("{married, never married} / {divorced, separated}", [(0, 3), (1, 2)]),
        ("{married, divorced, separated} / {never married}", [(0, 1, 2), (3,)]),
        ("{married, divorced} / {separated, never married}", [(0, 1), (2, 3)]),
    ]
    nested_rows = []
    for label, groups in candidate_groups:
        # Candidate nests test which marital statuses have correlated unobservables.
        fit = fit_nested(women_data, groups, start_base=women_fit["params"])
        nested_rows.append(
            {
                "grouping": label,
                "loglike": fit["loglike"],
                "taus": ", ".join(f"{tau:.3f}" for tau in fit["taus"]),
                "success": fit["success"],
            }
        )

    print("Exercise 26.12")
    print(f"men_loglike = {men_fit['loglike']:.6f}")
    print(men_grid.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("comparison_women_same_spec")
    print(f"women_loglike = {women_fit['loglike']:.6f}")
    print(women_grid.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Exercise 26.13")
    print(f"women_le_35_loglike = {women35_fit['loglike']:.6f}")
    print(women35_coef.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("average_marginal_effects")
    print(women35_ame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Exercise 26.14")
    print(pd.DataFrame(nested_rows).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
