"""Replicate Chapter 17 panel-data GMM exercises.

The script constructs Arellano-Bond and Blundell-Bond moment matrices by hand,
then prints dynamic panel estimates so the instrument design is inspectable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PYTHON_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PYTHON_ROOT / "data"


def as_float(value: str) -> float:
    """Parse tab-delimited numeric fields while treating dataset missing codes as NA."""
    text = value.strip()
    if text in {"", ".", "NA"}:
        return float("nan")
    return float(text)


def load_rows(name: str) -> list[dict[str, float | int]]:
    """Load a panel dataset from Hansen's tab-delimited text files."""
    path = DATA_ROOT / name / f"{name}.txt"
    with path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        rows: list[dict[str, float | int]] = []
        for row in reader:
            parsed: dict[str, float | int] = {}
            for key, value in row.items():
                assert value is not None
                if key in {"id", "cusip", "year"}:
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = as_float(value)
            rows.append(parsed)
    return rows


def group_panel(rows: list[dict[str, float | int]], id_key: str, names: Iterable[str]) -> list[dict[str, object]]:
    """Collect rows by firm and turn each variable into a time-ordered array."""
    grouped: dict[int, list[dict[str, float | int]]] = {}
    for row in rows:
        key = int(row[id_key])
        grouped.setdefault(key, []).append(row)

    panels: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: int(row["year"]))
        years = np.array([int(row["year"]) for row in group], dtype=int)
        panel: dict[str, object] = {"id": key, "year": years}
        for name in names:
            panel[name] = np.array([float(row[name]) for row in group], dtype=float)
        panels.append(panel)
    return panels


def symmetric_inverse(matrix: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Use a symmetric pseudo-inverse for potentially redundant GMM instruments."""
    matrix = 0.5 * (matrix + matrix.T)
    return np.linalg.pinv(matrix, rcond=rcond, hermitian=True)


def h_matrix(length: int) -> np.ndarray:
    """Return the first-difference error covariance pattern for one panel."""
    if length <= 0:
        return np.zeros((0, 0), dtype=float)
    h = 2.0 * np.eye(length)
    if length >= 2:
        off = np.ones(length - 1)
        h -= np.diag(off, 1)
        h -= np.diag(off, -1)
    return h


@dataclass(frozen=True)
class RegressorSpec:
    """Describe a panel regressor and its exogeneity status."""

    name: str
    lag: int
    kind: str
    label: str


@dataclass(frozen=True)
class ModelSpec:
    """Collect the choices that define an AB or BB dynamic panel estimator."""

    data_name: str
    id_key: str
    dep: str
    dep_lags: int
    regressors: tuple[RegressorSpec, ...]
    year_effects: bool = False
    estimator: str = "ab"
    steps: int = 1
    robust: bool = True
    lag_depth: int | None = None


@dataclass
class DiffBlock:
    """Column layout for instruments in one differenced equation row."""

    dep_start: int
    dep_count: int
    pred_starts: tuple[int, ...]
    pred_counts: tuple[int, ...]
    size: int


@dataclass
class LevelBlock:
    """Column layout for instruments in one system-GMM level equation row."""

    dep_start: int
    pred_start: int
    size: int


@dataclass
class Layout:
    """Global instrument-column bookkeeping shared by all panels."""

    t0: int
    m_max: int
    q_diff_gmm: int
    q_diff_iv_start: int
    q_diff: int
    q_level_gmm_start: int
    q_level_iv_start: int
    q_total: int
    diff_blocks: tuple[DiffBlock, ...]
    level_blocks: tuple[LevelBlock, ...]
    n_pred: int
    n_strict: int
    n_dummies: int


@dataclass
class PanelMatrices:
    """Stacked outcome, regressor, instrument, and covariance pieces for one firm."""

    y: np.ndarray
    x: np.ndarray
    z: np.ndarray
    h: np.ndarray


@dataclass
class GMMResult:
    """Estimated coefficients and reporting metadata for one GMM specification."""

    beta: np.ndarray
    se: np.ndarray
    covariance: np.ndarray
    instruments: int
    equation_rows: int
    firms: int


def build_year_dummies(years: np.ndarray, all_years: list[int]) -> np.ndarray:
    """Create year effects with the first year omitted as the base category."""
    if len(all_years) <= 1:
        return np.zeros((len(years), 0), dtype=float)
    kept = all_years[1:]
    return np.column_stack([(years == year).astype(float) for year in kept])


def valid_lag_indices(max_index: int, limit: int | None) -> list[int]:
    """List available lagged instruments, optionally truncating instrument depth."""
    if max_index < 0:
        return []
    if limit is None or limit >= max_index + 1:
        start = 0
    else:
        start = max_index + 1 - limit
    return list(range(start, max_index + 1))


def make_layout(t_max: int, spec: ModelSpec, n_dummies: int) -> Layout:
    """Precompute instrument-column positions for the selected GMM design."""
    max_reg_lag = max((reg.lag for reg in spec.regressors), default=0)
    t0 = max(spec.dep_lags + 1, max_reg_lag + 1)
    m_max = max(t_max - t0, 0)

    pred_regs = tuple(reg for reg in spec.regressors if reg.kind == "pred")
    strict_regs = tuple(reg for reg in spec.regressors if reg.kind == "strict")

    diff_blocks: list[DiffBlock] = []
    column = 0
    for row in range(m_max):
        t = t0 + row
        # Differenced equations use levels dated t-2 and earlier as instruments.
        dep_count = len(valid_lag_indices(t - 2, spec.lag_depth))
        dep_start = column
        column += dep_count

        pred_starts: list[int] = []
        pred_counts: list[int] = []
        for reg in pred_regs:
            # Predetermined regressors use lags that precede the differenced error.
            available = t - reg.lag - 1
            count = len(valid_lag_indices(available, spec.lag_depth))
            pred_starts.append(column)
            pred_counts.append(count)
            column += count

        diff_blocks.append(
            DiffBlock(
                dep_start=dep_start,
                dep_count=dep_count,
                pred_starts=tuple(pred_starts),
                pred_counts=tuple(pred_counts),
                size=column - dep_start,
            )
        )

    q_diff_gmm = column
    q_diff_iv_start = column
    column += len(strict_regs) + n_dummies
    q_diff = column

    level_blocks: list[LevelBlock] = []
    q_level_gmm_start = q_diff
    q_level_iv_start = q_diff
    if spec.estimator == "bb":
        if spec.dep_lags != 1:
            raise ValueError("System GMM support here is implemented only for AR(1) models.")
        q_level_gmm_start = column
        for _row in range(m_max):
            # Blundell-Bond adds level equations instrumented by lagged differences.
            dep_start = column
            column += 1
            pred_start = column
            column += len(pred_regs)
            level_blocks.append(
                LevelBlock(
                    dep_start=dep_start,
                    pred_start=pred_start,
                    size=column - dep_start,
                )
            )
        q_level_iv_start = column
        column += len(strict_regs) + n_dummies

    return Layout(
        t0=t0,
        m_max=m_max,
        q_diff_gmm=q_diff_gmm,
        q_diff_iv_start=q_diff_iv_start,
        q_diff=q_diff,
        q_level_gmm_start=q_level_gmm_start,
        q_level_iv_start=q_level_iv_start,
        q_total=column,
        diff_blocks=tuple(diff_blocks),
        level_blocks=tuple(level_blocks),
        n_pred=len(pred_regs),
        n_strict=len(strict_regs),
        n_dummies=n_dummies,
    )


def panel_matrices(panel: dict[str, object], spec: ModelSpec, layout: Layout, all_years: list[int]) -> PanelMatrices | None:
    """Build transformed equations and instruments for one firm."""
    years = np.asarray(panel["year"], dtype=int)
    y = np.asarray(panel[spec.dep], dtype=float)
    t_i = len(y)
    if t_i <= layout.t0:
        return None

    pred_regs = tuple(reg for reg in spec.regressors if reg.kind == "pred")
    strict_regs = tuple(reg for reg in spec.regressors if reg.kind == "strict")
    dummies = build_year_dummies(years, all_years) if spec.year_effects else np.zeros((t_i, 0), dtype=float)

    rows = list(range(layout.t0, t_i))
    m = len(rows)

    x_cols: list[np.ndarray] = []
    # The AB block works with first differences to remove firm fixed effects.
    for lag in range(1, spec.dep_lags + 1):
        x_cols.append(np.array([y[t - lag] - y[t - lag - 1] for t in rows], dtype=float))
    for reg in spec.regressors:
        series = np.asarray(panel[reg.name], dtype=float)
        x_cols.append(np.array([series[t - reg.lag] - series[t - reg.lag - 1] for t in rows], dtype=float))
    for j in range(layout.n_dummies):
        x_cols.append(np.array([dummies[t, j] - dummies[t - 1, j] for t in rows], dtype=float))
    x_diff = np.column_stack(x_cols) if x_cols else np.zeros((m, 0), dtype=float)
    y_diff = np.array([y[t] - y[t - 1] for t in rows], dtype=float)

    z_diff = np.zeros((m, layout.q_diff), dtype=float)
    for local_row, t in enumerate(rows):
        block = layout.diff_blocks[local_row]

        dep_indices = valid_lag_indices(t - 2, spec.lag_depth)
        if block.dep_count:
            z_diff[local_row, block.dep_start : block.dep_start + block.dep_count] = y[dep_indices]

        for reg_index, reg in enumerate(pred_regs):
            available = t - reg.lag - 1
            idx = valid_lag_indices(available, spec.lag_depth)
            if block.pred_counts[reg_index]:
                series = np.asarray(panel[reg.name], dtype=float)
                start = block.pred_starts[reg_index]
                z_diff[local_row, start : start + block.pred_counts[reg_index]] = series[idx]

    diff_iv_col = layout.q_diff_iv_start
    for reg in strict_regs:
        series = np.asarray(panel[reg.name], dtype=float)
        z_diff[:, diff_iv_col] = np.array([series[t - reg.lag] - series[t - reg.lag - 1] for t in rows], dtype=float)
        diff_iv_col += 1
    if layout.n_dummies:
        for j in range(layout.n_dummies):
            z_diff[:, diff_iv_col + j] = np.array([dummies[t, j] - dummies[t - 1, j] for t in rows], dtype=float)

    h_diff = h_matrix(m)

    if spec.estimator == "ab":
        return PanelMatrices(y=y_diff, x=x_diff, z=z_diff, h=h_diff)

    # The BB system appends level equations to recover information in persistent series.
    x_level_cols: list[np.ndarray] = []
    x_level_cols.append(np.array([y[t - 1] for t in rows], dtype=float))
    for reg in spec.regressors:
        series = np.asarray(panel[reg.name], dtype=float)
        x_level_cols.append(np.array([series[t - reg.lag] for t in rows], dtype=float))
    for j in range(layout.n_dummies):
        x_level_cols.append(np.array([dummies[t, j] for t in rows], dtype=float))
    x_level = np.column_stack(x_level_cols) if x_level_cols else np.zeros((m, 0), dtype=float)
    y_level = np.array([y[t] for t in rows], dtype=float)

    z_level = np.zeros((m, layout.q_total - layout.q_diff), dtype=float)
    for local_row, t in enumerate(rows):
        block = layout.level_blocks[local_row]
        z_level[local_row, block.dep_start - layout.q_diff] = y[t - 1] - y[t - 2]

        for reg_index, reg in enumerate(pred_regs):
            series = np.asarray(panel[reg.name], dtype=float)
            value = series[t - reg.lag] - series[t - reg.lag - 1]
            z_level[local_row, block.pred_start - layout.q_diff + reg_index] = value

    level_iv_col = layout.q_level_iv_start - layout.q_diff
    for reg in strict_regs:
        series = np.asarray(panel[reg.name], dtype=float)
        z_level[:, level_iv_col] = np.array([series[t - reg.lag] for t in rows], dtype=float)
        level_iv_col += 1
    if layout.n_dummies:
        for j in range(layout.n_dummies):
            z_level[:, level_iv_col + j] = np.array([dummies[t, j] for t in rows], dtype=float)

    y_stack = np.concatenate([y_diff, y_level])
    x_stack = np.vstack([x_diff, x_level])
    z_stack = np.zeros((2 * m, layout.q_total), dtype=float)
    z_stack[:m, : layout.q_diff] = z_diff
    z_stack[m:, layout.q_diff :] = z_level
    h_stack = np.block(
        [
            [h_diff, np.zeros((m, m), dtype=float)],
            [np.zeros((m, m), dtype=float), np.eye(m, dtype=float)],
        ]
    )
    return PanelMatrices(y=y_stack, x=x_stack, z=z_stack, h=h_stack)


def estimate_gmm(panels: list[dict[str, object]], spec: ModelSpec) -> GMMResult:
    """Estimate one- or two-step linear GMM for a dynamic panel specification."""
    all_years = sorted({int(year) for panel in panels for year in np.asarray(panel["year"], dtype=int)})
    t_max = max(len(np.asarray(panel[spec.dep], dtype=float)) for panel in panels)
    n_dummies = len(all_years) - 1 if spec.year_effects else 0
    layout = make_layout(t_max=t_max, spec=spec, n_dummies=n_dummies)

    k = spec.dep_lags + len(spec.regressors) + n_dummies
    xz = np.zeros((k, layout.q_total), dtype=float)
    zy = np.zeros(layout.q_total, dtype=float)
    omega1 = np.zeros((layout.q_total, layout.q_total), dtype=float)
    equation_rows = 0
    used_firms = 0

    for panel in panels:
        mats = panel_matrices(panel, spec, layout, all_years)
        if mats is None:
            continue
        used_firms += 1
        equation_rows += mats.y.size
        xz += mats.x.T @ mats.z
        zy += mats.z.T @ mats.y
        omega1 += mats.z.T @ mats.h @ mats.z

    # One-step GMM uses the known differenced-error covariance structure.
    w1 = symmetric_inverse(omega1)
    bread1 = xz @ w1 @ xz.T
    bread1_inv = symmetric_inverse(bread1)
    beta1 = bread1_inv @ (xz @ w1 @ zy)

    omega2 = np.zeros_like(omega1)
    gls_ssr_1 = 0.0
    for panel in panels:
        mats = panel_matrices(panel, spec, layout, all_years)
        if mats is None:
            continue
        residual = mats.y - mats.x @ beta1
        omega2 += mats.z.T @ np.outer(residual, residual) @ mats.z
        h_inv = symmetric_inverse(mats.h)
        gls_ssr_1 += float(residual.T @ h_inv @ residual)

    sigma2_1 = gls_ssr_1 / max(equation_rows - k, 1)
    if spec.robust:
        # The sandwich covariance uses residual moment variation as the meat.
        meat1 = xz @ w1 @ omega2 @ w1 @ xz.T
        covariance1 = bread1_inv @ meat1 @ bread1_inv
    else:
        covariance1 = sigma2_1 * bread1_inv

    beta = beta1
    covariance = covariance1

    if spec.steps == 2:
        # Two-step GMM updates the weighting matrix with first-step residuals.
        w2 = symmetric_inverse(omega2)
        bread2 = xz @ w2 @ xz.T
        bread2_inv = symmetric_inverse(bread2)
        beta2 = bread2_inv @ (xz @ w2 @ zy)

        omega3 = np.zeros_like(omega1)
        gls_ssr_2 = 0.0
        for panel in panels:
            mats = panel_matrices(panel, spec, layout, all_years)
            if mats is None:
                continue
            residual = mats.y - mats.x @ beta2
            omega3 += mats.z.T @ np.outer(residual, residual) @ mats.z
            h_inv = symmetric_inverse(mats.h)
            gls_ssr_2 += float(residual.T @ h_inv @ residual)

        sigma2_2 = gls_ssr_2 / max(equation_rows - k, 1)
        if spec.robust:
            meat2 = xz @ w2 @ omega3 @ w2 @ xz.T
            covariance2 = bread2_inv @ meat2 @ bread2_inv
        else:
            covariance2 = sigma2_2 * bread2_inv
        beta = beta2
        covariance = covariance2

    se = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    return GMMResult(
        beta=beta,
        se=se,
        covariance=covariance,
        instruments=layout.q_total,
        equation_rows=equation_rows,
        firms=used_firms,
    )


def coefficient_names(spec: ModelSpec, panels: list[dict[str, object]]) -> list[str]:
    """Match printed coefficient labels to the transformed design matrix."""
    years = sorted({int(year) for panel in panels for year in np.asarray(panel["year"], dtype=int)})
    names: list[str] = []
    for lag in range(1, spec.dep_lags + 1):
        names.append(f"L{lag}.{spec.dep}")
    names.extend(reg.label for reg in spec.regressors)
    if spec.year_effects:
        names.extend(f"year_{year}" for year in years[1:])
    return names


def print_result(title: str, spec: ModelSpec, panels: list[dict[str, object]]) -> None:
    """Estimate a specification and print coefficients with key design metadata."""
    result = estimate_gmm(panels, spec)
    names = coefficient_names(spec, panels)
    print(title)
    print(
        f"  firms={result.firms}, transformed_rows={result.equation_rows}, "
        f"instruments={result.instruments}, steps={spec.steps}, "
        f"estimator={spec.estimator}, robust={spec.robust}, lag_depth={spec.lag_depth}"
    )
    for name, beta, se in zip(names, result.beta, result.se):
        print(f"  {name:12s} {beta: .6f}  ({se:.6f})")
    print()


def ab1991_panels() -> list[dict[str, object]]:
    """Load the Arellano-Bond employment panel."""
    rows = load_rows("AB1991")
    return group_panel(rows, "id", ["n", "w", "k"])


def invest1993_panels() -> list[dict[str, object]]:
    """Load the investment panel used for dynamic debt specifications."""
    rows = load_rows("Invest1993")
    return group_panel(rows, "cusip", ["debta", "inva", "vala", "cfa"])


def run_ab1991() -> None:
    """Run the AB1991 employment and labor-demand specifications."""
    panels = ab1991_panels()

    print_result(
        "Exercise 17.15(a): AB one-step AR(1) for k with year effects",
        ModelSpec(
            data_name="AB1991",
            id_key="id",
            dep="k",
            dep_lags=1,
            regressors=(),
            year_effects=True,
            estimator="ab",
            steps=1,
            robust=True,
        ),
        panels,
    )

    print_result(
        "Exercise 17.15(b): BB one-step AR(1) for k with year effects",
        ModelSpec(
            data_name="AB1991",
            id_key="id",
            dep="k",
            dep_lags=1,
            regressors=(),
            year_effects=True,
            estimator="bb",
            steps=1,
            robust=True,
        ),
        panels,
    )

    strict_regs = (
        RegressorSpec("w", 0, "strict", "w"),
        RegressorSpec("w", 1, "strict", "L1.w"),
        RegressorSpec("k", 0, "strict", "k"),
        RegressorSpec("k", 1, "strict", "L1.k"),
    )
    mixed_pred_regs = (
        RegressorSpec("w", 0, "strict", "w"),
        RegressorSpec("w", 1, "pred", "L1.w"),
        RegressorSpec("k", 0, "strict", "k"),
        RegressorSpec("k", 1, "pred", "L1.k"),
    )

    # Strict and predetermined classifications change which lagged variables are valid instruments.
    print_result(
        "Exercise 17.16(a): AB one-step labor demand, w and k strictly exogenous",
        ModelSpec(
            data_name="AB1991",
            id_key="id",
            dep="n",
            dep_lags=1,
            regressors=strict_regs,
            year_effects=True,
            estimator="ab",
            steps=1,
            robust=True,
        ),
        panels,
    )

    print_result(
        "Exercise 17.16(b): AB one-step labor demand, lagged w and k predetermined, lag_depth=2",
        ModelSpec(
            data_name="AB1991",
            id_key="id",
            dep="n",
            dep_lags=1,
            regressors=mixed_pred_regs,
            year_effects=True,
            estimator="ab",
            steps=1,
            robust=True,
            lag_depth=2,
        ),
        panels,
    )

    print_result(
        "Exercise 17.16(c): BB one-step labor demand, lagged w and k predetermined",
        ModelSpec(
            data_name="AB1991",
            id_key="id",
            dep="n",
            dep_lags=1,
            regressors=mixed_pred_regs,
            year_effects=True,
            estimator="bb",
            steps=1,
            robust=True,
            lag_depth=2,
        ),
        panels,
    )

    print_result(
        "Exercise 17.16(e): BB one-step labor demand, classical SEs only",
        ModelSpec(
            data_name="AB1991",
            id_key="id",
            dep="n",
            dep_lags=1,
            regressors=mixed_pred_regs,
            year_effects=True,
            estimator="bb",
            steps=1,
            robust=False,
            lag_depth=2,
        ),
        panels,
    )


def run_invest1993() -> None:
    """Run the Invest1993 dynamic capital-structure specifications."""
    panels = invest1993_panels()

    print_result(
        "Exercise 17.17(a): AB two-step AR(1) for debta",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=(),
            estimator="ab",
            steps=2,
            robust=True,
        ),
        panels,
    )

    print_result(
        "Exercise 17.17(b): BB two-step AR(1) for debta",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=(),
            estimator="bb",
            steps=2,
            robust=True,
        ),
        panels,
    )

    print_result(
        "Exercise 17.17(c1): AB one-step AR(1) for debta",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=(),
            estimator="ab",
            steps=1,
            robust=True,
        ),
        panels,
    )

    print_result(
        "Exercise 17.17(c2): AB two-step AR(2) for debta",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=2,
            regressors=(),
            estimator="ab",
            steps=2,
            robust=True,
        ),
        panels,
    )

    print_result(
        "Exercise 17.17(c3): AB two-step AR(1), lag_depth=2",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=(),
            estimator="ab",
            steps=2,
            robust=True,
            lag_depth=2,
        ),
        panels,
    )

    print_result(
        "Exercise 17.17(c4): AB two-step AR(1), classical SEs",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=(),
            estimator="ab",
            steps=2,
            robust=False,
        ),
        panels,
    )

    pred_regs = (
        RegressorSpec("inva", 1, "pred", "L1.inva"),
        RegressorSpec("vala", 1, "pred", "L1.vala"),
        RegressorSpec("cfa", 1, "pred", "L1.cfa"),
    )

    print_result(
        "Exercise 17.18(a): AB two-step, all regressors predetermined, lag_depth=2",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=pred_regs,
            estimator="ab",
            steps=2,
            robust=True,
            lag_depth=2,
        ),
        panels,
    )

    print_result(
        "Exercise 17.18(b): BB two-step, all regressors predetermined, lag_depth=2",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=pred_regs,
            estimator="bb",
            steps=2,
            robust=True,
            lag_depth=2,
        ),
        panels,
    )

    print_result(
        "Exercise 17.18(c1): AB one-step, all regressors predetermined",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=pred_regs,
            estimator="ab",
            steps=1,
            robust=True,
            lag_depth=2,
        ),
        panels,
    )

    print_result(
        "Exercise 17.18(c2): AB two-step, all regressors predetermined, lag_depth=2",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=pred_regs,
            estimator="ab",
            steps=2,
            robust=True,
            lag_depth=2,
        ),
        panels,
    )

    print_result(
        "Exercise 17.18(c3): AB two-step, all regressors predetermined, classical SEs, lag_depth=2",
        ModelSpec(
            data_name="Invest1993",
            id_key="cusip",
            dep="debta",
            dep_lags=1,
            regressors=pred_regs,
            estimator="ab",
            steps=2,
            robust=False,
            lag_depth=2,
        ),
        panels,
    )


def main() -> None:
    run_ab1991()
    run_invest1993()


if __name__ == "__main__":
    main()
