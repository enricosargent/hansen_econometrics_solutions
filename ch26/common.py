"""Shared discrete-choice machinery for Chapter 26 exercises.

The helpers convert datasets into choice arrays and estimate multinomial logit,
nested logit, mixed logit, and multinomial probit models with explicit normalizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, ndtr, ndtri
from scipy.stats import qmc
from statsmodels.tools.numdiff import approx_hess


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.data_loader import load_dataset


MARITAL_LABELS = ["married", "divorced", "separated", "never married"]
KOPPELMAN_ALTS = ["train", "air", "bus", "car"]


@dataclass
class ChoiceData:
    """Container for long-form choice data reshaped into estimator-ready arrays."""

    y: np.ndarray
    x_common: np.ndarray
    w_case: np.ndarray
    alt_names: list[str]
    case_feature_names: list[str]
    common_feature_names: list[str]

    @property
    def n(self) -> int:
        """Number of observed choice situations."""
        return int(self.y.shape[0])

    @property
    def j(self) -> int:
        """Number of alternatives in each choice set."""
        return int(len(self.alt_names))

    @property
    def p_common(self) -> int:
        """Number of alternative-varying regressors with common slopes."""
        return int(self.x_common.shape[2]) if self.x_common.ndim == 3 else 0

    @property
    def q_case(self) -> int:
        """Number of case-specific regressors with alternative-specific slopes."""
        return int(self.w_case.shape[1]) if self.w_case.ndim == 2 else 0


def add_intercept(matrix: np.ndarray) -> np.ndarray:
    """Prepend a constant column for alternative- or case-specific regressors."""
    return np.column_stack([np.ones(matrix.shape[0]), matrix])


def positive_part(values: np.ndarray) -> np.ndarray:
    """Return the hinge term used in spline bases."""
    return np.maximum(values, 0.0)


def quadratic_spline(age: np.ndarray, knots: tuple[float, ...]) -> np.ndarray:
    """Build age, age squared, and squared-hinge terms for marital-status logits."""
    columns = [age, age**2]
    for knot in knots:
        columns.append(positive_part(age - knot) ** 2)
    return np.column_stack(columns)


def marital_four_category(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse CPS marital codes into Hansen's four modeled alternatives."""
    data = frame.copy()
    data = data.loc[data["marital"].isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])].copy()
    conditions = [
        data["marital"].isin([1.0, 2.0, 3.0, 4.0]),
        data["marital"].eq(5.0),
        data["marital"].eq(6.0),
        data["marital"].eq(7.0),
    ]
    data["marital4"] = np.select(conditions, [0, 1, 2, 3], default=np.nan).astype(int)
    return data


def load_cps_marital() -> pd.DataFrame:
    """Load CPS data and attach the four-category marital choice outcome."""
    return marital_four_category(load_dataset("cps09mar"))


def load_koppelman_long() -> pd.DataFrame:
    """Load Koppelman's travel-mode data and standardize alternative indices."""
    data = load_dataset("Koppelman").copy()
    alternative = data["alternative"]
    if pd.api.types.is_numeric_dtype(alternative):
        data["alt_index"] = alternative.astype(int) - 1
    else:
        mapping = {name: index for index, name in enumerate(KOPPELMAN_ALTS)}
        data["alt_index"] = alternative.astype(str).str.strip().str.lower().map(mapping)
        if data["alt_index"].isna().any():
            unknown = sorted(data.loc[data["alt_index"].isna(), "alternative"].astype(str).unique())
            raise ValueError(f"Unknown Koppelman alternatives: {unknown}")
        data["alt_index"] = data["alt_index"].astype(int)
    return data


def wide_koppelman(
    common_vars: list[str],
    case_vars: list[str] | None = None,
) -> ChoiceData:
    """Reshape Koppelman's long alternative rows into one choice set per case."""
    data = load_koppelman_long()
    cases = np.sort(data["case"].unique())
    n = len(cases)
    j = len(KOPPELMAN_ALTS)
    x_common = np.zeros((n, j, len(common_vars)), float)
    w_case = np.zeros((n, len(case_vars or [])), float)
    y = np.zeros(n, int)

    by_case = data.sort_values(["case", "alt_index"]).groupby("case", sort=True)
    for idx, (_, group) in enumerate(by_case):
        group = group.sort_values("alt_index")
        x_common[idx] = group[common_vars].to_numpy(float)
        if case_vars:
            w_case[idx] = group.iloc[0][case_vars].to_numpy(float)
        y[idx] = int(group.loc[group["choice"].eq(1), "alt_index"].iloc[0])

    return ChoiceData(
        y=y,
        x_common=x_common,
        w_case=w_case,
        alt_names=KOPPELMAN_ALTS.copy(),
        case_feature_names=list(case_vars or []),
        common_feature_names=common_vars,
    )


def unpack_mnl_params(
    params: np.ndarray,
    n_alt: int,
    p_common: int,
    q_case: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split MNL parameters into common slopes and case-specific alternative effects."""
    gamma = params[:p_common]
    beta = np.zeros((n_alt, q_case), float)
    # Alternative 0 is the normalization: its case-specific coefficients are zero.
    if q_case:
        beta[1:] = params[p_common:].reshape(n_alt - 1, q_case)
    return gamma, beta


def utilities_from_params(params: np.ndarray, data: ChoiceData) -> np.ndarray:
    """Evaluate systematic utility for every observation and alternative."""
    gamma, beta = unpack_mnl_params(params, data.j, data.p_common, data.q_case)
    utilities = np.zeros((data.n, data.j), float)
    if data.p_common:
        utilities += np.einsum("njp,p->nj", data.x_common, gamma)
    if data.q_case:
        utilities += data.w_case @ beta.T
    return utilities


def softmax_probabilities(utilities: np.ndarray) -> np.ndarray:
    """Convert utilities to logit probabilities with a numerical stabilization."""
    stabilized = utilities - utilities.max(axis=1, keepdims=True)
    exp_u = np.exp(stabilized)
    return exp_u / exp_u.sum(axis=1, keepdims=True)


def mnl_loglike_obs(params: np.ndarray, data: ChoiceData) -> np.ndarray:
    """Return each observation's log probability of the chosen alternative."""
    probabilities = softmax_probabilities(utilities_from_params(params, data))
    return np.log(np.clip(probabilities[np.arange(data.n), data.y], 1e-300, None))


def mnl_negloglike(params: np.ndarray, data: ChoiceData) -> float:
    """Objective minimized by the generic MNL optimizer."""
    return float(-mnl_loglike_obs(params, data).sum())


def mnl_gradient(params: np.ndarray, data: ChoiceData) -> np.ndarray:
    """Analytic MNL score, written as chosen indicators minus probabilities."""
    utilities = utilities_from_params(params, data)
    probabilities = softmax_probabilities(utilities)
    indicators = np.zeros_like(probabilities)
    indicators[np.arange(data.n), data.y] = 1.0
    residual = indicators - probabilities
    pieces: list[np.ndarray] = []
    if data.p_common:
        grad_gamma = np.einsum("nj,njp->p", residual, data.x_common)
        pieces.append(grad_gamma)
    if data.q_case:
        grad_beta = np.zeros((data.j - 1, data.q_case), float)
        for alt in range(1, data.j):
            grad_beta[alt - 1] = residual[:, alt] @ data.w_case
        pieces.append(grad_beta.ravel())
    gradient = np.concatenate(pieces) if pieces else np.zeros(0, float)
    return -gradient


def fit_mnl(data: ChoiceData, start: np.ndarray | None = None) -> dict[str, object]:
    """Estimate multinomial logit, using statsmodels when its design matches."""
    if data.p_common == 0 and data.q_case:
        model = sm.MNLogit(data.y, data.w_case)
        result = model.fit(
            start_params=None if start is None else np.asarray(start, dtype=float),
            method="newton",
            maxiter=400,
            disp=False,
        )
        params = np.asarray(result.params, dtype=float).ravel(order="F")
        se = np.asarray(result.bse, dtype=float).ravel(order="F")
        return {
            "params": params,
            "se": se,
            "cov": np.asarray(result.cov_params(), dtype=float),
            "loglike": float(result.llf),
            "success": bool(getattr(result, "mle_retvals", {}).get("converged", True)),
            "message": "statsmodels MNLogit converged"
            if getattr(result, "mle_retvals", {}).get("converged", True)
            else "statsmodels MNLogit did not report convergence",
            "result": result,
        }

    k = data.p_common + (data.j - 1) * data.q_case
    x0 = np.zeros(k, float) if start is None else start.astype(float).copy()
    result = minimize(
        mnl_negloglike,
        x0,
        args=(data,),
        jac=mnl_gradient,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 400},
    )
    hess = approx_hess(result.x, lambda p: mnl_negloglike(p, data))
    covariance = np.linalg.pinv(0.5 * (hess + hess.T))
    return {
        "params": result.x,
        "se": np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        "cov": covariance,
        "loglike": -mnl_negloglike(result.x, data),
        "success": bool(result.success),
        "message": result.message,
        "result": result,
    }


def mnl_probabilities(params: np.ndarray, data: ChoiceData) -> np.ndarray:
    """Convenience wrapper for fitted MNL choice probabilities."""
    return softmax_probabilities(utilities_from_params(params, data))


def group_map(groups: list[tuple[int, ...]]) -> dict[int, int]:
    """Map each alternative index to its nest index."""
    mapping: dict[int, int] = {}
    for group_index, group in enumerate(groups):
        for alt in group:
            mapping[alt] = group_index
    return mapping


def unpack_nested_params(
    params: np.ndarray,
    data: ChoiceData,
    groups: list[tuple[int, ...]],
    fixed_taus: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover base utility parameters and nest dissimilarity parameters."""
    fixed_taus = fixed_taus or {}
    base_dim = data.p_common + (data.j - 1) * data.q_case
    base_params = params[:base_dim]
    tau_params = params[base_dim:]
    taus = np.ones(len(groups), float)
    cursor = 0
    for group_index, group in enumerate(groups):
        if len(group) == 1:
            taus[group_index] = 1.0
        elif group_index in fixed_taus:
            taus[group_index] = fixed_taus[group_index]
        else:
            taus[group_index] = expit(tau_params[cursor])
            cursor += 1
    return base_params, taus


def nested_logprob_matrix(
    params: np.ndarray,
    data: ChoiceData,
    groups: list[tuple[int, ...]],
    fixed_taus: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute nested-logit log probabilities from within- and across-nest pieces."""
    base_params, taus = unpack_nested_params(params, data, groups, fixed_taus)
    utilities = utilities_from_params(base_params, data)
    log_prob = np.full_like(utilities, -np.inf)
    group_logs = []
    group_indices = group_map(groups)
    within_logs: list[np.ndarray] = []

    for tau, group in zip(taus, groups):
        subset = utilities[:, list(group)] / tau
        log_i = logsumexp(subset, axis=1)
        group_logs.append(tau * log_i)
        within_logs.append(subset - log_i[:, None])

    log_denom = logsumexp(np.column_stack(group_logs), axis=1)
    for group_index, group in enumerate(groups):
        log_pg = group_logs[group_index] - log_denom
        log_prob[:, list(group)] = within_logs[group_index] + log_pg[:, None]
    return log_prob, taus


def nested_negloglike(
    params: np.ndarray,
    data: ChoiceData,
    groups: list[tuple[int, ...]],
    fixed_taus: dict[int, float] | None = None,
) -> float:
    """Objective for nested-logit maximum likelihood."""
    log_prob, _ = nested_logprob_matrix(params, data, groups, fixed_taus)
    return float(-log_prob[np.arange(data.n), data.y].sum())


def fit_nested(
    data: ChoiceData,
    groups: list[tuple[int, ...]],
    fixed_taus: dict[int, float] | None = None,
    start_base: np.ndarray | None = None,
    start_tau: list[float] | None = None,
) -> dict[str, object]:
    """Estimate nested logit with free tau values constrained to the unit interval."""
    fixed_taus = fixed_taus or {}
    base_dim = data.p_common + (data.j - 1) * data.q_case
    free_groups = [
        idx
        for idx, group in enumerate(groups)
        if len(group) > 1 and idx not in fixed_taus
    ]
    tau_start = np.array(start_tau or [0.7] * len(free_groups), float)
    tau_raw = np.log(tau_start / (1.0 - tau_start))
    x0 = np.concatenate(
        [
            np.zeros(base_dim, float) if start_base is None else start_base.astype(float),
            tau_raw,
        ]
    )
    objective = lambda p: nested_negloglike(p, data, groups, fixed_taus)
    result = minimize(objective, x0, method="BFGS", options={"gtol": 1e-6, "maxiter": 400})
    hess = approx_hess(result.x, objective)
    covariance = np.linalg.pinv(0.5 * (hess + hess.T))
    _, taus = unpack_nested_params(result.x, data, groups, fixed_taus)
    return {
        "params": result.x,
        "se": np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        "cov": covariance,
        "taus": taus,
        "loglike": -objective(result.x),
        "success": bool(result.success),
        "message": result.message,
        "result": result,
    }


def unpack_mixed_params(
    params: np.ndarray,
    data: ChoiceData,
    p_fixed: int,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Split mixed-logit parameters into fixed slopes and a random coefficient law."""
    gamma_fixed = params[:p_fixed]
    mu = float(params[p_fixed])
    sigma = float(np.exp(params[p_fixed + 1]))
    beta = np.zeros((data.j, data.q_case), float)
    if data.q_case:
        beta[1:] = params[p_fixed + 2 :].reshape(data.j - 1, data.q_case)
    return gamma_fixed, mu, sigma, beta


def hermite_rule(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Hermite nodes and weights for integrating normal heterogeneity."""
    nodes, weights = hermgauss(n_nodes)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def mixed_probabilities(
    params: np.ndarray,
    data: ChoiceData,
    x_random: np.ndarray,
    p_fixed: int,
    n_nodes: int,
    distribution: str,
) -> np.ndarray:
    """Integrate logit probabilities over the random coefficient distribution."""
    nodes, weights = hermite_rule(n_nodes)
    gamma_fixed, mu, sigma, beta = unpack_mixed_params(params, data, p_fixed)
    utilities = np.zeros((data.n, data.j), float)
    if p_fixed:
        utilities += np.einsum("njp,p->nj", data.x_common[:, :, :p_fixed], gamma_fixed)
    if data.q_case:
        utilities += data.w_case @ beta.T

    integrated = np.zeros((data.n, data.j), float)
    for node, weight in zip(nodes, weights):
        if distribution == "normal":
            coefficient = mu + sigma * node
        elif distribution == "neg_lognormal":
            coefficient = -np.exp(mu + sigma * node)
        else:
            raise ValueError(distribution)
        draw_utilities = utilities + coefficient * x_random
        integrated += weight * softmax_probabilities(draw_utilities)
    return integrated


def mixed_negloglike(
    params: np.ndarray,
    data: ChoiceData,
    x_random: np.ndarray,
    p_fixed: int,
    n_nodes: int,
    distribution: str,
) -> float:
    """Objective for simulated mixed-logit likelihood."""
    probabilities = mixed_probabilities(params, data, x_random, p_fixed, n_nodes, distribution)
    chosen = np.clip(probabilities[np.arange(data.n), data.y], 1e-300, None)
    return float(-np.log(chosen).sum())


def fit_mixed(
    data: ChoiceData,
    x_random: np.ndarray,
    p_fixed: int,
    n_nodes: int,
    distribution: str,
    start: np.ndarray,
) -> dict[str, object]:
    """Estimate mixed logit by minimizing the quadrature-integrated likelihood."""
    objective = lambda p: mixed_negloglike(p, data, x_random, p_fixed, n_nodes, distribution)
    result = minimize(
        objective,
        start.astype(float),
        method="BFGS",
        options={"gtol": 1e-5, "maxiter": 300},
    )
    covariance = np.asarray(result.hess_inv)
    return {
        "params": result.x,
        "se": np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        "cov": covariance,
        "loglike": -objective(result.x),
        "success": bool(result.success),
        "message": result.message,
        "result": result,
    }


MNP_A = {
    0: np.eye(3),
    1: np.array([[-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]),
    2: np.array([[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, -1.0, 1.0]]),
    3: np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
}


def unpack_mnp_params(params: np.ndarray, data: ChoiceData) -> tuple[np.ndarray, np.ndarray]:
    """Recover MNP utility parameters and a positive-definite covariance matrix."""
    base_dim = data.p_common + (data.j - 1) * data.q_case
    base_params = params[:base_dim]
    a21, log_d22, a31, a32, log_d33 = params[base_dim:]
    lower = np.array(
        [
            [np.sqrt(2.0), 0.0, 0.0],
            [a21, np.exp(log_d22), 0.0],
            [a31, a32, np.exp(log_d33)],
        ]
    )
    sigma = lower @ lower.T
    return base_params, sigma


def halton_draws(n_obs: int, n_draws: int, dim: int, seed: int = 0) -> np.ndarray:
    """Generate quasi-random draws used by the GHK simulator."""
    engine = qmc.Halton(d=dim, scramble=True, seed=seed)
    draws = engine.random(n_obs * n_draws).reshape(n_draws, n_obs, dim)
    return np.clip(draws, 1e-8, 1.0 - 1e-8)


def ghk_probability(mu: np.ndarray, cov: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Approximate multivariate normal orthant probabilities by GHK simulation."""
    cov = 0.5 * (cov + cov.T)
    try:
        lower = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigenvalues = np.linalg.eigvalsh(cov)
        if not np.all(np.isfinite(eigenvalues)):
            raise
        jitter = max(1e-10, float(-eigenvalues.min()) + 1e-10)
        lower = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
    u1 = draws[:, :, 0]
    a1 = (-mu[:, 0] / lower[0, 0])[None, :]
    p1 = np.clip(ndtr(a1), 1e-12, 1.0)
    z1 = ndtri(u1 * p1)

    u2 = draws[:, :, 1]
    a2 = (-(mu[:, 1][None, :] + lower[1, 0] * z1) / lower[1, 1])
    p2 = np.clip(ndtr(a2), 1e-12, 1.0)
    z2 = ndtri(u2 * p2)

    u3 = draws[:, :, 2]
    a3 = (
        -(
            mu[:, 2][None, :]
            + lower[2, 0] * z1
            + lower[2, 1] * z2
        )
        / lower[2, 2]
    )
    p3 = np.clip(ndtr(a3), 1e-12, 1.0)
    return (p1 * p2 * p3).mean(axis=0)


def mnp_choice_probabilities(
    params: np.ndarray,
    data: ChoiceData,
    draws: np.ndarray,
) -> np.ndarray:
    """Compute simulated MNP probabilities for each observed chosen alternative."""
    base_params, sigma = unpack_mnp_params(params, data)
    utilities = utilities_from_params(base_params, data)
    mean_diff = np.column_stack(
        [
            utilities[:, 1] - utilities[:, 0],
            utilities[:, 2] - utilities[:, 0],
            utilities[:, 3] - utilities[:, 0],
        ]
    )
    probabilities = np.zeros(data.n, float)
    for alt in range(4):
        mask = data.y == alt
        if not np.any(mask):
            continue
        matrix = MNP_A[alt]
        transformed_mean = mean_diff[mask] @ matrix.T
        transformed_cov = matrix @ sigma @ matrix.T
        probabilities[mask] = ghk_probability(transformed_mean, transformed_cov, draws[:, mask, :])
    return probabilities


def mnp_negloglike(params: np.ndarray, data: ChoiceData, draws: np.ndarray) -> float:
    """Penalized negative log likelihood for numerically delicate MNP fits."""
    try:
        probabilities = np.clip(mnp_choice_probabilities(params, data, draws), 1e-300, None)
        value = float(-np.log(probabilities).sum())
    except (FloatingPointError, ValueError, np.linalg.LinAlgError, OverflowError):
        return 1e12
    if not np.isfinite(value):
        return 1e12
    return value


def fit_mnp(data: ChoiceData, draws: np.ndarray, start: np.ndarray) -> dict[str, object]:
    """Estimate multinomial probit with simulated probabilities."""
    objective = lambda p: mnp_negloglike(p, data, draws)
    result = minimize(
        objective,
        start.astype(float),
        method="BFGS",
        options={"gtol": 1e-4, "maxiter": 250},
    )
    covariance = np.asarray(result.hess_inv)
    return {
        "params": result.x,
        "se": np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        "cov": covariance,
        "loglike": -objective(result.x),
        "success": bool(result.success),
        "message": result.message,
        "result": result,
    }


def covariance_to_mnp_moments(sigma: np.ndarray) -> dict[str, float]:
    """Summarize an MNP covariance matrix as variances and correlations."""
    var_air = sigma[0, 0]
    var_bus = sigma[1, 1]
    var_car = sigma[2, 2]
    corr_air_bus = sigma[0, 1] / np.sqrt(var_air * var_bus)
    corr_air_car = sigma[0, 2] / np.sqrt(var_air * var_car)
    corr_bus_car = sigma[1, 2] / np.sqrt(var_bus * var_car)
    return {
        "var_air": float(var_air),
        "var_bus": float(var_bus),
        "var_car": float(var_car),
        "corr_air_bus": float(corr_air_bus),
        "corr_air_car": float(corr_air_car),
        "corr_bus_car": float(corr_bus_car),
    }
