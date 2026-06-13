"""Replicate Chapter 12 instrumental-variables empirical exercises.

The script works through AJR, Card, and Angrist-Krueger examples to show how
2SLS, LIML, control functions, and bootstrap IV inference are computed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS, IVLIML
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from python.ch10.ch10_utils import bc_interval
from python.data_loader import load_dataset

AJR_BOOTSTRAP_REPS = 10_000
AJR_BOOTSTRAP_SEED_1 = 1223
AJR_BOOTSTRAP_SEED_2 = 2223
CARD_BOOTSTRAP_REPS = 3_000
CARD_BOOTSTRAP_SEED = 1225
AK_BOOTSTRAP_REPS = 300
AK_BOOTSTRAP_SEED = 1227


def format_number(value: float) -> str:
    """Format scalar output consistently across the printed exercise tables."""
    return f"{float(value):.6f}"


def residualize(vector: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Remove the linear projection on controls for Frisch-Waugh-Lovell IV steps."""
    result = sm.OLS(np.asarray(vector, float).reshape(-1), np.asarray(controls, float)).fit()
    return np.asarray(result.resid, dtype=float)



def residualize_matrix(matrix: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residualize each instrument column against the included controls."""
    matrix = np.asarray(matrix, float)
    if matrix.ndim == 1:
        return residualize(matrix, controls)[:, None]
    return np.column_stack([residualize(matrix[:, column], controls) for column in range(matrix.shape[1])])


def residualize_qr(vector: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residualize with a QR projection for bootstrap draws that may be ill-conditioned."""
    q, _ = np.linalg.qr(np.asarray(controls, float))
    return np.asarray(vector, float) - q @ (q.T @ np.asarray(vector, float))


def residualize_matrix_qr(matrix: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Apply the QR residual maker to every column of an instrument matrix."""
    q, _ = np.linalg.qr(np.asarray(controls, float))
    return np.asarray(matrix, float) - q @ (q.T @ np.asarray(matrix, float))



def select_independent_columns(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Drop redundant instruments after residualization using pivoted QR rank."""
    qz, rz, piv = qr(np.asarray(matrix, float), mode="economic", pivoting=True)
    del qz
    rank = int((np.abs(np.diag(rz)) > tol).sum())
    keep = np.sort(piv[:rank])
    return np.asarray(matrix, float)[:, keep]


def scalar_iv_closed_form(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Compute scalar 2SLS and robust standard error from moment matrices."""
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[:, None]
    n = y.size
    qzz = z.T @ z / n
    qzx = z.T @ x / n
    qxz = x @ z / n
    qzy = z.T @ y / n
    beta = float((qxz @ np.linalg.solve(qzz, qzy)) / (qxz @ np.linalg.solve(qzz, qzx)))
    u = y - x * beta
    omega = (z.T * (u**2)) @ z / n
    a = float(qxz @ np.linalg.solve(qzz, qzx))
    b = float(qxz @ np.linalg.solve(qzz, omega) @ np.linalg.solve(qzz, qzx))
    variance = b / (a**2)
    return beta, float(np.sqrt(variance / n))



def scalar_iv_robust(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Delegate scalar IV estimation to linearmodels for package-verified results."""
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[:, None]
    fit = IV2SLS(y, None, pd.DataFrame({"x": x}), pd.DataFrame(z)).fit(cov_type="robust")
    return float(fit.params.iloc[0]), float(fit.std_errors.iloc[0])



def scalar_iv_fwl(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    controls: np.ndarray,
    *,
    select_independent: bool = False,
    use_package: bool = True,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a one-endogenous-regressor IV model after partialling controls."""
    if use_package:
        yr = residualize(y, controls)
        xr = residualize(x, controls)
        zr = residualize_matrix(z, controls)
    else:
        yr = residualize_qr(y, controls)
        xr = residualize_qr(x, controls)
        zr = residualize_matrix_qr(z, controls)
    if select_independent:
        zr = select_independent_columns(zr)
    beta, se = scalar_iv_robust(yr, xr, zr) if use_package else scalar_iv_closed_form(yr, xr, zr)
    return beta, se, yr, xr, zr



def bootstrap_scalar_iv(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    controls: np.ndarray,
    *,
    reps: int,
    seed: int,
    select_independent: bool = False,
) -> np.ndarray:
    """Pairs-bootstrap the scalar IV coefficient, skipping singular resamples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    estimates = np.empty(reps)
    filled = 0
    while filled < reps:
        index = rng.integers(0, n, size=n)
        try:
            beta, _, _, _, _ = scalar_iv_fwl(
                y[index],
                x[index],
                z[index],
                controls[index],
                select_independent=select_independent,
                use_package=False,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        if np.isfinite(beta):
            estimates[filled] = beta
            filled += 1
    return estimates



def ajr_results() -> None:
    """Run AJR colonial-origins IV specifications and bootstrap checks."""
    # Construct the transformations used to compare log and level mortality instruments.
    ajr = load_dataset("AJR2001").copy()
    ajr["const"] = 1.0
    ajr["logmort0_sq"] = ajr["logmort0"] ** 2
    ajr["mort0"] = np.exp(ajr["logmort0"])

    y = ajr["loggdp"]
    # OLS, reduced form, and 2SLS are printed together to show the IV chain.
    ols = sm.OLS(y, ajr[["const", "risk"]]).fit()
    ols_robust = sm.OLS(y, ajr[["const", "risk"]]).fit(cov_type="HC1")

    reduced = sm.OLS(ajr["risk"], ajr[["const", "logmort0"]]).fit()
    reduced_robust = sm.OLS(ajr["risk"], ajr[["const", "logmort0"]]).fit(cov_type="HC1")

    iv = IV2SLS(y, ajr[["const"]], ajr[["risk"]], ajr[["logmort0"]]).fit(cov_type="unadjusted")
    iv_robust = IV2SLS(y, ajr[["const"]], ajr[["risk"]], ajr[["logmort0"]]).fit(cov_type="robust")

    pi = sm.OLS(y, ajr[["const", "logmort0"]]).fit().params["logmort0"]
    lam = reduced.params["logmort0"]
    ils_beta = float(pi / lam)

    # The control-function regression includes the first-stage residual directly.
    control_function = sm.OLS(
        y,
        sm.add_constant(pd.DataFrame({"risk": ajr["risk"], "u_hat": reduced.resid})),
    ).fit(cov_type="HC1")

    ols_controls = sm.OLS(y, ajr[["const", "risk", "latitude", "africa"]]).fit(cov_type="HC1")
    iv_controls = IV2SLS(
        y,
        ajr[["const", "latitude", "africa"]],
        ajr[["risk"]],
        ajr[["logmort0"]],
    ).fit(cov_type="robust")

    reduced_level = sm.OLS(ajr["risk"], ajr[["const", "mort0"]]).fit(cov_type="HC1")

    reduced_sq = sm.OLS(ajr["risk"], ajr[["const", "logmort0", "logmort0_sq"]]).fit()
    reduced_sq_test = reduced_sq.f_test("logmort0 = 0, logmort0_sq = 0")
    iv_sq = IV2SLS(y, ajr[["const"]], ajr[["risk"]], ajr[["logmort0", "logmort0_sq"]]).fit(cov_type="robust")
    liml_sq = IVLIML(y, ajr[["const"]], ajr[["risk"]], ajr[["logmort0", "logmort0_sq"]]).fit(cov_type="robust")

    y_np = ajr["loggdp"].to_numpy(float)
    x_np = ajr["risk"].to_numpy(float)
    z_np = ajr["logmort0"].to_numpy(float)[:, None]
    controls_np = np.ones((len(ajr), 1))
    beta_exact, _, _, _, _ = scalar_iv_fwl(y_np, x_np, z_np, controls_np)
    # Fixed seeds make the resampling comparison reproducible.
    bootstrap_1 = bootstrap_scalar_iv(
        y_np,
        x_np,
        z_np,
        controls_np,
        reps=AJR_BOOTSTRAP_REPS,
        seed=AJR_BOOTSTRAP_SEED_1,
    )
    bootstrap_2 = bootstrap_scalar_iv(
        y_np,
        x_np,
        z_np,
        controls_np,
        reps=AJR_BOOTSTRAP_REPS,
        seed=AJR_BOOTSTRAP_SEED_2,
    )

    print("Exercise 12.22")
    print("baseline_ols")
    print(ols.params.to_string())
    print("homoskedastic_se")
    print(ols.bse.to_string())
    print("robust_se")
    print(ols_robust.bse.to_string())
    print()
    print("reduced_form")
    print(reduced.params.to_string())
    print("homoskedastic_se")
    print(reduced.bse.to_string())
    print("robust_se")
    print(reduced_robust.bse.to_string())
    print(f"first_stage_F = {format_number(float(reduced.tvalues['logmort0'] ** 2))}")
    print()
    print("iv_2sls")
    print(iv.params.to_string())
    print("homoskedastic_se")
    print(iv.std_errors.to_string())
    print("robust_se")
    print(iv_robust.std_errors.to_string())
    print(f"indirect_least_squares_beta = {format_number(ils_beta)}")
    print(f"two_stage_beta = {format_number(float(IV2SLS(y, ajr[['const']], ajr[['risk']], ajr[['logmort0']]).fit(cov_type='robust').params['risk']))}")
    print("control_function")
    print(control_function.params.to_string())
    print(control_function.bse.to_string())
    print()
    print("ols_with_latitude_africa")
    print(ols_controls.params.to_string())
    print(ols_controls.bse.to_string())
    print("iv_with_latitude_africa")
    print(iv_controls.params.to_string())
    print(iv_controls.std_errors.to_string())
    print()
    print("reduced_form_level_mortality")
    print(reduced_level.params.to_string())
    print(reduced_level.bse.to_string())
    print(f"r2_log = {format_number(float(reduced.rsquared))}")
    print(f"r2_level = {format_number(float(sm.OLS(ajr['risk'], ajr[['const', 'mort0']]).fit().rsquared))}")
    print()
    print("quadratic_first_stage")
    print(reduced_sq.params.to_string())
    print(reduced_sq.bse.to_string())
    print(f"joint_F = {format_number(float(reduced_sq_test.fvalue))}")
    print("iv_two_instruments")
    print(iv_sq.params.to_string())
    print(iv_sq.std_errors.to_string())
    print(f"sargan_stat = {format_number(float(iv_sq.sargan.stat))}")
    print(f"sargan_p = {format_number(float(iv_sq.sargan.pval))}")
    print("liml_two_instruments")
    print(liml_sq.params.to_string())
    print(liml_sq.std_errors.to_string())
    print()
    print("Exercise 12.23")
    print(f"beta = {format_number(beta_exact)}")
    print(f"bootstrap_se_seed_{AJR_BOOTSTRAP_SEED_1} = {format_number(float(bootstrap_1.std(ddof=1)))}")
    print(f"bootstrap_bc_seed_{AJR_BOOTSTRAP_SEED_1} = {bc_interval(bootstrap_1, beta_exact)}")
    print(f"bootstrap_se_seed_{AJR_BOOTSTRAP_SEED_2} = {format_number(float(bootstrap_2.std(ddof=1)))}")
    print(f"bootstrap_bc_seed_{AJR_BOOTSTRAP_SEED_2} = {bc_interval(bootstrap_2, beta_exact)}")
    print()



def card_results() -> None:
    """Run Card proximity-to-college IV specifications and bootstrap checks."""
    card = load_dataset("Card1995").copy()
    numeric_columns = [
        "lwage76",
        "age76",
        "ed76",
        "nearc2",
        "nearc4",
        "nearc4a",
        "nearc4b",
        "black",
        "reg76r",
        "smsa76r",
    ]
    for column in numeric_columns:
        card[column] = pd.to_numeric(card[column], errors="coerce").astype(float)
    # Experience controls follow the Mincer-style wage specification in the text.
    card["exp"] = card["age76"] - card["ed76"] - 6
    card["exp2"] = card["exp"] ** 2 / 100.0
    card["age2"] = card["age76"] ** 2 / 100.0
    card = card.loc[card["lwage76"].notna()].copy().reset_index(drop=True)
    card["const"] = 1.0
    card["nearc4a_age76"] = card["nearc4a"] * card["age76"]
    card["nearc4a_age2"] = card["nearc4a"] * card["age2"]

    baseline_controls = ["const", "exp", "exp2", "black", "reg76r", "smsa76r"]
    y = card["lwage76"]

    # Public/private college proximity instruments identify schooling variation.
    reduced = sm.OLS(card["ed76"], card[baseline_controls + ["nearc4a", "nearc4b"]]).fit(cov_type="HC1")
    iv = IV2SLS(y, card[baseline_controls], card[["ed76"]], card[["nearc4a", "nearc4b"]]).fit(cov_type="robust")

    reduced_nearc2 = sm.OLS(card["ed76"], card[baseline_controls + ["nearc4a", "nearc4b", "nearc2"]]).fit(cov_type="HC1")

    reduced_interactions = sm.OLS(
        card["ed76"],
        card[baseline_controls + ["nearc4a", "nearc4b", "nearc4a_age76", "nearc4a_age2"]],
    ).fit(cov_type="HC1")
    reduced_interactions_classical = sm.OLS(
        card["ed76"],
        card[baseline_controls + ["nearc4a", "nearc4b", "nearc4a_age76", "nearc4a_age2"]],
    ).fit()
    interaction_f_test = reduced_interactions_classical.f_test(
        "nearc4a = 0, nearc4b = 0, nearc4a_age76 = 0, nearc4a_age2 = 0"
    )

    iv_expanded = IV2SLS(
        y,
        card[baseline_controls],
        card[["ed76"]],
        card[["nearc4a", "nearc4b", "nearc4a_age76", "nearc4a_age2"]],
    ).fit(cov_type="robust")
    liml_expanded = IVLIML(
        y,
        card[baseline_controls],
        card[["ed76"]],
        card[["nearc4a", "nearc4b", "nearc4a_age76", "nearc4a_age2"]],
    ).fit(cov_type="robust")

    first_stage_cf = sm.OLS(
        card["ed76"],
        card[baseline_controls + ["nearc4a", "nearc4b", "nearc4a_age76", "nearc4a_age2"]],
    ).fit()
    card["u_hat"] = first_stage_cf.resid
    # Adding the first-stage residual is the control-function diagnostic.
    control_function = sm.OLS(
        y,
        card[["const", "ed76", "exp", "exp2", "black", "reg76r", "smsa76r", "u_hat"]],
    ).fit(cov_type="HC1")

    reduced_single = sm.OLS(card["ed76"], card[baseline_controls + ["nearc4"]]).fit(cov_type="HC1")
    iv_single = IV2SLS(y, card[baseline_controls], card[["ed76"]], card[["nearc4"]]).fit(cov_type="robust")

    y_np = card["lwage76"].to_numpy(float)
    x_np = card["ed76"].to_numpy(float)
    z_np = card["nearc4"].to_numpy(float)[:, None]
    controls_np = card[["const", "exp", "exp2", "black", "reg76r", "smsa76r"]].to_numpy(float)
    beta_single, _, _, _, _ = scalar_iv_fwl(y_np, x_np, z_np, controls_np)
    bootstrap_single = bootstrap_scalar_iv(
        y_np,
        x_np,
        z_np,
        controls_np,
        reps=CARD_BOOTSTRAP_REPS,
        seed=CARD_BOOTSTRAP_SEED,
    )

    print("Exercise 12.24")
    print("reduced_form_public_private")
    print(reduced.params.to_string())
    print(reduced.bse.to_string())
    print("iv_public_private")
    print(iv.params.to_string())
    print(iv.std_errors.to_string())
    print()
    print("reduced_form_add_nearc2")
    print(reduced_nearc2.params.to_string())
    print(reduced_nearc2.bse.to_string())
    print()
    print("reduced_form_interactions")
    print(reduced_interactions.params.to_string())
    print(reduced_interactions.bse.to_string())
    print(f"expanded_first_stage_F = {format_number(float(interaction_f_test.fvalue))}")
    print("iv_expanded")
    print(iv_expanded.params.to_string())
    print(iv_expanded.std_errors.to_string())
    print("liml_expanded")
    print(liml_expanded.params.to_string())
    print(liml_expanded.std_errors.to_string())
    print("control_function")
    print(control_function.params.to_string())
    print(control_function.bse.to_string())
    print(control_function.pvalues.to_string())
    print()
    print("Exercise 12.25")
    print("reduced_form_single_instrument")
    print(reduced_single.params.to_string())
    print(reduced_single.bse.to_string())
    print("iv_single_instrument")
    print(iv_single.params.to_string())
    print(iv_single.std_errors.to_string())
    print(f"bootstrap_beta = {format_number(beta_single)}")
    print(f"bootstrap_se = {format_number(float(bootstrap_single.std(ddof=1)))}")
    print(f"bootstrap_bc = {bc_interval(bootstrap_single, beta_single)}")
    print()



def ak_black_results() -> None:
    """Run Angrist-Krueger quarter-of-birth IV specifications for Black men."""
    ak = load_dataset("AK1991").copy()
    blk = ak.loc[ak["black"].eq(1)].copy().reset_index(drop=True)

    # Quarter-of-birth instruments are interacted with cohort and state indicators.
    y = blk["logwage"].to_numpy(float)
    x = blk["edu"].to_numpy(float)
    qob3 = pd.get_dummies(blk["qob"].astype(int), prefix="qob", drop_first=True, dtype=float)
    yob9 = pd.get_dummies(blk["yob"].astype(int), prefix="yob", drop_first=True, dtype=float)
    yob10 = pd.get_dummies(blk["yob"].astype(int), prefix="yobf", drop_first=False, dtype=float)
    region8 = pd.get_dummies(blk["region"].astype(int), prefix="region", drop_first=True, dtype=float)
    state50 = pd.get_dummies(blk["state"].astype(int), prefix="state", drop_first=True, dtype=float)

    qob3_np = qob3.to_numpy(float)
    yob9_np = yob9.to_numpy(float)
    region8_np = region8.to_numpy(float)
    state50_np = state50.to_numpy(float)

    qob_yob = np.column_stack(
        [
            qob3_np[:, iq] * yob10.iloc[:, iy].to_numpy(float)
            for iq in range(qob3_np.shape[1])
            for iy in range(yob10.shape[1])
        ]
    )
    qob_state = np.column_stack(
        [
            qob3_np[:, iq] * state50.iloc[:, is_].to_numpy(float)
            for iq in range(qob3_np.shape[1])
            for is_ in range(state50.shape[1])
        ]
    )

    controls_many = np.column_stack(
        [
            np.ones(len(blk)),
            blk["smsa"].to_numpy(float),
            blk["married"].to_numpy(float),
            yob9_np,
            region8_np,
            state50_np,
        ]
    )
    controls_small = np.column_stack(
        [
            np.ones(len(blk)),
            blk["smsa"].to_numpy(float),
            blk["married"].to_numpy(float),
            yob9_np,
            region8_np,
        ]
    )

    z_many = np.column_stack([qob_yob, qob_state])
    z_30 = qob_yob
    z_3 = qob3_np

    beta_many, se_many, yr_many, xr_many, zr_many = scalar_iv_fwl(
        y,
        x,
        z_many,
        controls_many,
        select_independent=True,
    )
    # First-stage F statistics are computed on the residualized first-stage regression.
    f_many = sm.OLS(xr_many, zr_many).fit().fvalue
    p_many = sm.OLS(xr_many, zr_many).fit().f_pvalue

    beta_30, se_30, yr_30, xr_30, zr_30 = scalar_iv_fwl(
        y,
        x,
        z_30,
        controls_small,
        select_independent=True,
    )
    f_30_fit = sm.OLS(xr_30, zr_30).fit()

    beta_3, se_3, yr_3, xr_3, zr_3 = scalar_iv_fwl(y, x, z_3, controls_small)
    f_3_fit = sm.OLS(xr_3, zr_3).fit()
    liml_3 = IVLIML(
        yr_3,
        None,
        pd.DataFrame({"edu": xr_3}),
        pd.DataFrame(zr_3),
    ).fit(cov_type="robust")

    bootstrap_3 = bootstrap_scalar_iv(
        y,
        x,
        z_3,
        controls_small,
        reps=AK_BOOTSTRAP_REPS,
        seed=AK_BOOTSTRAP_SEED,
    )

    print("Exercise 12.26")
    print("analog_12_90_black_men")
    print(f"excluded_instrument_rank = {zr_many.shape[1]}")
    print(f"first_stage_F = {format_number(float(f_many))}")
    print(f"first_stage_p = {format_number(float(p_many))}")
    print(f"beta_many = {format_number(beta_many)}")
    print(f"se_many = {format_number(se_many)}")
    print()
    print("analog_12_89_black_men")
    print(f"first_stage_F = {format_number(float(f_30_fit.fvalue))}")
    print(f"first_stage_p = {format_number(float(f_30_fit.f_pvalue))}")
    print(f"beta_30 = {format_number(beta_30)}")
    print(f"se_30 = {format_number(se_30)}")
    print()
    print("analog_12_92_black_men")
    print(f"first_stage_F = {format_number(float(f_3_fit.fvalue))}")
    print(f"first_stage_p = {format_number(float(f_3_fit.f_pvalue))}")
    print(f"beta_3 = {format_number(beta_3)}")
    print(f"se_3 = {format_number(se_3)}")
    print(f"liml_beta_3 = {format_number(float(liml_3.params['edu']))}")
    print(f"liml_se_3 = {format_number(float(liml_3.std_errors['edu']))}")
    print()
    print("Exercise 12.27")
    print(f"bootstrap_se = {format_number(float(bootstrap_3.std(ddof=1)))}")
    print(f"bootstrap_bc = {bc_interval(bootstrap_3, beta_3)}")
    print()



def main() -> None:
    ajr_results()
    card_results()
    ak_black_results()


if __name__ == "__main__":
    main()
