"""Shared nonlinear-regression helpers for Chapter 23 exercises.

The helpers load workbook data, estimate nonlinear least-squares covariance
matrices, and format output for CES and threshold-regression examples.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PYTHON_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PYTHON_ROOT / "data"


def load_workbook_rows(relative_path):
    """Load workbook rows and a column-name index for formula-style scripts."""
    frame = pd.read_excel(DATA_ROOT / relative_path)
    header = list(frame.columns)
    index = {name: column for column, name in enumerate(header)}
    return list(frame.itertuples(index=False, name=None)), index


def to_float(value):
    """Convert workbook entries to floats while keeping missing values explicit."""
    if value in (None, ".") or pd.isna(value):
        return None
    return float(value)


def ols_fit(design, outcome):
    """Fit the linear least-squares step used inside nonlinear comparisons."""
    result = sm.OLS(np.asarray(outcome, dtype=float), np.asarray(design, dtype=float)).fit()
    beta = np.asarray(result.params, dtype=float)
    residual = np.asarray(result.resid, dtype=float)
    sse = float(result.ssr)
    return beta, residual, sse


def central_difference_jacobian(mean_function, params, steps):
    """Approximate the derivative of the fitted mean for sandwich inference."""
    base = np.asarray(mean_function(params), float)
    sample_size = len(base)
    parameter_count = len(params)
    jacobian = np.empty((sample_size, parameter_count))
    for column in range(parameter_count):
        step = steps[column] * max(1.0, abs(params[column]))
        plus = params.copy()
        minus = params.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            np.asarray(mean_function(plus), float) - np.asarray(mean_function(minus), float)
        ) / (plus[column] - minus[column])
    return jacobian


def sandwich_covariance(jacobian, residual, cluster=None):
    """Build heteroskedastic or cluster-robust covariance from nonlinear scores."""
    sample_size = len(residual)
    q_matrix = (jacobian.T @ jacobian) / sample_size
    q_inv = np.linalg.inv(q_matrix)

    if cluster is None:
        omega = (jacobian.T * (residual ** 2)) @ jacobian / sample_size
    else:
        omega = np.zeros((jacobian.shape[1], jacobian.shape[1]))
        for group in np.unique(cluster):
            mask = cluster == group
            score = jacobian[mask].T @ residual[mask]
            omega += np.outer(score, score)
        omega /= sample_size

    asymptotic_v = q_inv @ omega @ q_inv
    covariance = asymptotic_v / sample_size
    return covariance


def format_vector(values, digits=6):
    """Format coefficient vectors without changing numeric precision upstream."""
    rounded = [f"{float(value):.{digits}f}" for value in values]
    return "[" + ", ".join(rounded) + "]"


def logsumexp_two(a, b, weight_a, weight_b):
    """Compute a stable weighted log-sum-exp for smooth maximum expressions."""
    maximum = np.maximum(a, b)
    return maximum + np.log(weight_a * np.exp(a - maximum) + weight_b * np.exp(b - maximum))


def logistic(value):
    """Evaluate the logistic function without overflow in either tail."""
    if value >= 0:
        exp_minus = math.exp(-value)
        return 1.0 / (1.0 + exp_minus)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
