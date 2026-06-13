"""Replicate Exercise 16.14 with Johansen cointegration tests.

The code prepares paired macro series, computes trace statistics under several
deterministic specifications, and reports ranks for long-run relationships.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import _endog_matrices, _sij


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


CASES = [
    {
        "pair": "(tb3ms, gs10)",
        "columns": ["tb3ms", "gs10"],
        "transform": None,
        "var_trend": "c",
        "deterministic": "ci",
        "trend_model": "Trend Model 2",
        "cv0_5pct": 20.3,
        "cv1_5pct": 9.19,
    },
    {
        "pair": "(aaa, baa)",
        "columns": ["aaa", "baa"],
        "transform": None,
        "var_trend": "c",
        "deterministic": "ci",
        "trend_model": "Trend Model 2",
        "cv0_5pct": 20.3,
        "cv1_5pct": 9.19,
    },
    {
        "pair": "(log(ipdcongd), log(ipncongd))",
        "columns": ["ipdcongd", "ipncongd"],
        "transform": np.log,
        "var_trend": "ct",
        "deterministic": "coli",
        "trend_model": "Trend Model 4",
        "cv0_5pct": 25.9,
        "cv1_5pct": 12.5,
    },
]


def prepare_pair(frame: pd.DataFrame, columns: list[str], transform) -> pd.DataFrame:
    """Select a pair of series and apply logs where the cointegration case requires."""
    pair = frame[columns].astype(float)
    if transform is not None:
        pair = transform(pair)
    return pd.DataFrame(pair, columns=columns).dropna()


def johansen_trace(data: pd.DataFrame, deterministic: str, p: int) -> tuple[int, np.ndarray, list[float]]:
    """Compute Johansen trace statistics from statsmodels' VECM matrices."""
    endog = data.to_numpy().T
    _, delta_y_1_T, y_lag1, delta_x = _endog_matrices(
        endog=endog,
        exog=None,
        exog_coint=None,
        diff_lags=p - 1,
        deterministic=deterministic,
        seasons=0,
        first_season=0,
    )
    _, _, _, _, _, eigenvalues, _ = _sij(delta_x, delta_y_1_T, y_lag1)
    m = endog.shape[0]
    eigenvalues = np.real(eigenvalues[:m])
    nobs = y_lag1.shape[1]
    trace_stats = [
        float(-nobs * np.sum(np.log(1.0 - eigenvalues[r:])))
        for r in range(m)
    ]
    return nobs, eigenvalues, trace_stats


def main() -> None:
    frame = load_dataset("FRED-MD").copy()
    rows: list[dict[str, object]] = []

    # Each case fixes the deterministic terms before selecting the VAR lag length.
    for case in CASES:
        data = prepare_pair(frame, case["columns"], case["transform"])
        p = int(VAR(data).select_order(maxlags=12, trend=case["var_trend"]).selected_orders["aic"])
        nobs, eigenvalues, trace_stats = johansen_trace(data, case["deterministic"], p)
        rows.append(
            {
                "pair": case["pair"],
                "trend_model": case["trend_model"],
                "p": p,
                "n_eff": nobs,
                "lambda1": float(eigenvalues[0]),
                "lambda2": float(eigenvalues[1]),
                "LR(0)": trace_stats[0],
                "cv0_5pct": case["cv0_5pct"],
                "reject_H0_r0": trace_stats[0] > case["cv0_5pct"],
                "LR(1)": trace_stats[1],
                "cv1_5pct": case["cv1_5pct"],
                "reject_H0_r1": trace_stats[1] > case["cv1_5pct"],
            }
        )

    table = pd.DataFrame(rows)
    print("Exercise 16.14")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    ip_data = prepare_pair(frame, ["ipdcongd", "ipncongd"], np.log)
    ip_p = int(VAR(ip_data).select_order(maxlags=12, trend="ct").selected_orders["aic"])
    _, _, ip_trace_tm3 = johansen_trace(ip_data, "co", ip_p)
    print()
    print("IP pair under Trend Model 3")
    print(
        pd.DataFrame(
            [
                {
                    "p": ip_p,
                    "LR(0)": ip_trace_tm3[0],
                    "cv0_5pct": 15.5,
                    "LR(1)": ip_trace_tm3[1],
                    "cv1_5pct": 3.85,
                }
            ]
        ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
