"""Replicate Chapter 14 time-series regression exercises.

The script estimates autoregressions, lag regressions, and impulse responses so
serial dependence, HAC inference, and dynamic effects are visible in code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


def ar_design(series: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Create the dependent variable and lag matrix for an AR(order) regression."""
    y = series[order:]
    columns = [np.ones(len(y))]
    for lag in range(1, order + 1):
        columns.append(series[order - lag : len(series) - lag])
    return y, np.column_stack(columns)


def ar_fit(series: np.ndarray, order: int, cov_type: str = "HC1", maxlags: int | None = None):
    """Fit an autoregression and attach the requested robust covariance estimator."""
    y, x = ar_design(series, order)
    result = sm.OLS(y, x).fit()
    if cov_type == "HAC":
        return result.get_robustcov_results(cov_type="HAC", maxlags=maxlags or order)
    if cov_type == "HC1":
        return result.get_robustcov_results(cov_type="HC1")
    return result


def impulse_responses(ar_coefficients: np.ndarray, horizon: int) -> np.ndarray:
    """Recursively trace the response of an AR process to a one-unit innovation."""
    responses = [1.0]
    for step in range(1, horizon + 1):
        value = 0.0
        for lag, coefficient in enumerate(ar_coefficients, start=1):
            if step - lag >= 0:
                value += coefficient * responses[step - lag]
        responses.append(value)
    return np.asarray(responses[1:])


def lag_regression(
    frame: pd.DataFrame,
    y_variable: str,
    x_variable: str,
    y_lags: list[int],
    x_lags: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build distributed-lag and ARDL designs from named time-series columns."""
    y = frame[y_variable].astype(float).to_numpy()
    x = frame[x_variable].astype(float).to_numpy()
    max_lag = max([0] + y_lags + x_lags)
    mask = np.isfinite(y)
    for lag in y_lags:
        mask &= np.r_[np.repeat(False, lag), np.isfinite(y[:-lag])]
    for lag in x_lags:
        mask &= np.r_[np.repeat(False, lag), np.isfinite(x[:-lag])]
    mask[:max_lag] = False
    rows = np.where(mask)[0]

    columns = [np.ones(len(rows))]
    names = ["constant"]
    for lag in y_lags:
        columns.append(y[rows - lag])
        names.append(f"L{lag}.{y_variable}")
    for lag in x_lags:
        columns.append(x[rows - lag])
        names.append(f"L{lag}.{x_variable}")
    return y[rows], np.column_stack(columns), names


def print_coefficients(title: str, names: list[str], params: np.ndarray, se: np.ndarray) -> None:
    """Print coefficient tables with standard errors in the manual's compact style."""
    print(title)
    for name, coefficient, standard_error in zip(names, params, se):
        print(f"  {name:16s} {coefficient: .6f}  ({standard_error:.6f})")
    print()


def main() -> None:
    # FRED-QD supplies quarterly series; FRED-MD supplies monthly unemployment.
    qd = load_dataset("FRED-QD")
    md = load_dataset("FRED-MD")

    pnfix_growth = 400.0 * np.diff(np.log(qd["pnfix"].to_numpy(float)))
    # HC1 and Newey-West SEs answer different serial-correlation assumptions.
    ar4_hc = ar_fit(pnfix_growth, 4, "HC1")
    ar4_nw = ar_fit(pnfix_growth, 4, "HAC", maxlags=5)
    names = ["constant", "L1", "L2", "L3", "L4"]
    print_coefficients("Exercise 14.18 AR(4), HC1 standard errors", names, ar4_hc.params, ar4_hc.bse)
    print("Exercise 14.18 Newey-West standard errors, M=5")
    for name, coefficient, standard_error in zip(names, ar4_nw.params, ar4_nw.bse):
        print(f"  {name:16s} {coefficient: .6f}  ({standard_error:.6f})")
    print("Exercise 14.18 impulse responses j=1,...,10")
    print("  " + " ".join(f"{value:.6f}" for value in impulse_responses(ar4_hc.params[1:], 10)))
    print()

    oil_change = np.diff(qd["oilpricex"].to_numpy(float))
    oil_change = oil_change[np.isfinite(oil_change)]
    oil_ar4 = ar_fit(oil_change, 4, "HC1")
    restriction = np.eye(4, 5, k=1)
    oil_wald = oil_ar4.wald_test(restriction, scalar=True)
    print_coefficients("Exercise 14.19 oil price first-difference AR(4), HC1", names, oil_ar4.params, oil_ar4.bse)
    print(f"Exercise 14.19 Wald test of four AR coefficients: statistic={float(oil_wald.statistic):.6f}, p={float(oil_wald.pvalue):.6f}")
    print()

    md["time"] = pd.to_datetime(md["time"])
    monthly_unrate = md.loc[md["time"] >= pd.Timestamp("1960-01-01"), "unrate"].astype(float).to_numpy()
    aic_rows = []
    # AIC is computed on a common sample so lag orders are comparable.
    for order in range(1, 9):
        y = monthly_unrate[8:]
        columns = [np.ones(len(y))]
        for lag in range(1, order + 1):
            columns.append(monthly_unrate[8 - lag : len(monthly_unrate) - lag])
        x = np.column_stack(columns)
        result = sm.OLS(y, x).fit()
        sse = float(result.resid @ result.resid)
        aic = len(y) * np.log(sse / len(y)) + 2.0 * x.shape[1]
        aic_rows.append((order, aic, result))
    print("Exercise 14.20 AIC by monthly AR order")
    for order, aic, _ in aic_rows:
        print(f"  AR({order}) {aic:.6f}")
    best_order, _, best_result = min(aic_rows, key=lambda row: row[1])
    best_hc = best_result.get_robustcov_results(cov_type="HC1")
    print_coefficients(
        f"Exercise 14.20 selected AR({best_order}), HC1 standard errors",
        ["constant"] + [f"L{lag}" for lag in range(1, best_order + 1)],
        best_hc.params,
        best_hc.bse,
    )

    for title, y_lags, x_lags in [
        ("Exercise 14.21 distributed lag", [], [1, 2, 3, 4]),
        ("Exercise 14.21 ARDL", [1, 2, 3, 4], [1, 2, 3, 4]),
    ]:
        y, x, reg_names = lag_regression(qd, "unrate", "claimsx", y_lags, x_lags)
        result = sm.OLS(y, x).fit().get_robustcov_results(cov_type="HAC", maxlags=4)
        print_coefficients(title, reg_names, result.params, result.bse)
        if y_lags:
            # Jointly testing lagged x terms is the Granger-causality check.
            restriction = np.zeros((4, len(reg_names)))
            for row, lag in enumerate(x_lags):
                restriction[row, reg_names.index(f"L{lag}.claimsx")] = 1.0
            test = result.wald_test(restriction, scalar=True)
            print(f"  Granger-causality Wald statistic={float(test.statistic):.6f}, p={float(test.pvalue):.6g}")
            print()

    qd_growth = qd.copy()
    qd_growth["gdpgrowth"] = np.r_[np.nan, 400.0 * np.diff(np.log(qd["gdpc1"].astype(float)))]
    for title, y_lags, x_lags in [
        ("Exercise 14.22 distributed lag", [], [1, 2, 3, 4]),
        ("Exercise 14.22 ARDL", [1, 2], [1, 2, 3, 4]),
    ]:
        y, x, reg_names = lag_regression(qd_growth, "gdpgrowth", "houst", y_lags, x_lags)
        result = sm.OLS(y, x).fit().get_robustcov_results(cov_type="HAC", maxlags=4)
        print_coefficients(title, reg_names, result.params, result.bse)
        if y_lags:
            restriction = np.zeros((4, len(reg_names)))
            for row, lag in enumerate(x_lags):
                restriction[row, reg_names.index(f"L{lag}.houst")] = 1.0
            test = result.wald_test(restriction, scalar=True)
            print(f"  Granger-causality Wald statistic={float(test.statistic):.6f}, p={float(test.pvalue):.6g}")
            print()


if __name__ == "__main__":
    main()
