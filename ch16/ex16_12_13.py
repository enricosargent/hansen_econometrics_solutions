"""Replicate Exercises 16.12-16.13 with unit-root and stationarity tests.

The script prepares macro time series and reports ADF and KPSS diagnostics so
students can compare the null hypotheses behind nonstationarity tests.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


MAX_P = 12
MAX_DF_LAGS = MAX_P - 1
KPSS_M = 26

ADF_CRIT_5 = {"c": -2.86, "ct": -3.41}
KPSS_CRIT_5 = {"c": 0.462, "ct": 0.148}

SERIES = [
    ("log(rpi)", "rpi", np.log, "ct"),
    ("indpro", "indpro", None, "ct"),
    ("houst", "houst", None, "c"),
    ("hwi", "hwi", None, "ct"),
    ("clf16ov", "clf16ov", None, "ct"),
    ("claims", "claimsx", None, "c"),
    ("ipfuels", "ipfuels", None, "ct"),
]


def prepare_series(frame: pd.DataFrame, column: str, transform) -> pd.Series:
    """Apply the exercise transformation and remove missing time-series entries."""
    series = frame[column].astype(float)
    if transform is not None:
        series = transform(series)
    return pd.Series(series).dropna()


def adf_result(series: pd.Series, regression: str) -> dict[str, float | int]:
    """Run an ADF test, retaining the selected lag order and coefficient estimate."""
    adf_stat, _, _, store = adfuller(
        series,
        maxlag=MAX_DF_LAGS,
        regression=regression,
        autolag="AIC",
        store=True,
        regresults=True,
    )
    return {
        "n": int(store.nobs),
        "p": int(store.usedlag + 1),
        "rho_hat_minus_1": float(store.resols.params[0]),
        "se": float(store.resols.bse[0]),
        "adf": float(adf_stat),
    }


def kpss_result(series: pd.Series, regression: str) -> dict[str, float | int]:
    """Run the KPSS stationarity test with Hansen's fixed truncation lag."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, _, nlags, _ = kpss(series, regression=regression, nlags=KPSS_M)
    return {"M": int(nlags), "kpss": float(stat)}


def main() -> None:
    frame = load_dataset("FRED-MD").copy()
    rows: list[dict[str, object]] = []

    # ADF has a unit-root null; KPSS has a stationarity null, so both decisions matter.
    for label, column, transform, regression in SERIES:
        series = prepare_series(frame, column, transform)
        adf = adf_result(series, regression)
        kpss_stats = kpss_result(series, regression)
        rows.append(
            {
                "series": label,
                "data_column": column,
                "spec": regression,
                "n": adf["n"],
                "p": adf["p"],
                "rho_hat_minus_1": adf["rho_hat_minus_1"],
                "se": adf["se"],
                "adf": adf["adf"],
                "adf_cv_5pct": ADF_CRIT_5[regression],
                "adf_reject_5pct": adf["adf"] < ADF_CRIT_5[regression],
                "M": kpss_stats["M"],
                "kpss": kpss_stats["kpss"],
                "kpss_cv_5pct": KPSS_CRIT_5[regression],
                "kpss_reject_5pct": kpss_stats["kpss"] > KPSS_CRIT_5[regression],
            }
        )

    table = pd.DataFrame(rows)
    print("Exercises 16.12-16.13")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
