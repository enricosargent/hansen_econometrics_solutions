"""Replicate Exercise 21.5 with a rectangular-kernel RDD estimate.

The script filters observations around the cutoff, fits local constants, and
prints treatment-effect estimates for several bandwidths.
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
OUTCOME = "mort_age59_related_postHS"


def rectangular_rdd(data: pd.DataFrame, bandwidth: float) -> tuple[int, float, float]:
    """Estimate the cutoff jump using observations inside a symmetric window."""
    sample = data.loc[(data["povrate60"] - CUTOFF).abs().le(bandwidth)].copy()
    x = sample["povrate60"].astype(float).to_numpy()
    centered = x - CUTOFF
    treatment = (x >= CUTOFF).astype(float)
    # The treatment coefficient is the discontinuity at the cutoff.
    design = np.column_stack(
        [
            np.ones(len(sample)),
            x,
            centered * treatment,
            treatment,
        ]
    )
    y = sample[OUTCOME].astype(float).to_numpy()
    fit = sm.OLS(y, design).fit()
    return len(sample), float(fit.params[3]), float(fit.bse[3])


def main() -> None:
    data = load_dataset("LM2007").copy()
    rows = []
    # Bandwidth sensitivity is the central diagnostic in this simple RDD estimate.
    for bandwidth in [13.8, 7.0, 20.0]:
        n, theta, se = rectangular_rdd(data, bandwidth)
        rows.append({"bandwidth": bandwidth, "n": n, "theta_hat": theta, "se_ols": se})

    print("Exercise 21.5")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.9f}"))


if __name__ == "__main__":
    main()
