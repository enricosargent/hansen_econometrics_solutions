"""Replicate Exercises 21.6-21.9 with local-linear RDD estimates.

The code applies triangular weights on each side of the cutoff and reports how
bandwidth and outcome choices affect the estimated discontinuity.
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


CUTOFF = 59.1984


def triangular_weights(x: np.ndarray, bandwidth: float) -> np.ndarray:
    """Apply Hansen's triangular RDD kernel with compact support."""
    u = (x - CUTOFF) / bandwidth
    weights = np.zeros_like(u, dtype=float)
    support = np.abs(u) < np.sqrt(6.0)
    weights[support] = (1.0 / np.sqrt(6.0)) * (1.0 - np.abs(u[support]) / np.sqrt(6.0))
    return weights


def local_linear_rdd(data: pd.DataFrame, outcome: str, bandwidth: float) -> tuple[int, float, float]:
    """Estimate a weighted local-linear RDD and return the cutoff jump."""
    x_all = data["povrate60"].astype(float).to_numpy()
    weights_all = triangular_weights(x_all, bandwidth)
    sample = data.loc[weights_all > 0].copy()
    weights = weights_all[weights_all > 0]

    x = sample["povrate60"].astype(float).to_numpy()
    centered = x - CUTOFF
    treatment = (x >= CUTOFF).astype(float)
    # Separate slopes on either side are created by the centered-by-treatment term.
    design = np.column_stack(
        [
            np.ones(len(sample)),
            centered,
            treatment,
            centered * treatment,
        ]
    )
    y = sample[outcome].astype(float).to_numpy()
    fit = sm.WLS(y, design, weights=weights).fit(cov_type="HC2")
    return len(sample), float(fit.params[2]), float(fit.bse[2])


def main() -> None:
    data = load_dataset("LM2007").copy()

    bandwidth_rows = []
    # First vary the bandwidth for the main Head Start mortality outcome.
    for bandwidth in [8.0, 4.0, 12.0]:
        n, theta, se = local_linear_rdd(data, "mort_age59_related_postHS", bandwidth)
        bandwidth_rows.append(
            {"bandwidth": bandwidth, "n": n, "theta_hat": theta, "se_hc2": se}
        )

    outcome_rows = []
    outcomes = {
        "mort_age59_injury_postHS": "injury mortality, ages 5-9, post Head Start",
        "mort_age25plus_related_postHS": "HS-related mortality, ages 25+, post Head Start",
        "mort_age59_related_preHS": "HS-related mortality, ages 5-9, pre Head Start",
    }
    for outcome, description in outcomes.items():
        n, theta, se = local_linear_rdd(data, outcome, 8.0)
        outcome_rows.append(
            {
                "outcome": outcome,
                "description": description,
                "n": n,
                "theta_hat": theta,
                "se_hc2": se,
            }
        )

    print("Exercise 21.6")
    print(pd.DataFrame(bandwidth_rows).to_string(index=False, float_format=lambda value: f"{value:.9f}"))
    print()
    print("Exercises 21.7-21.9")
    print(pd.DataFrame(outcome_rows).to_string(index=False, float_format=lambda value: f"{value:.9f}"))


if __name__ == "__main__":
    main()
