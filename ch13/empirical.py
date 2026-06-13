"""Replicate Chapter 13 GMM exercises for instrumental variables.

The code compares package IV-GMM estimates across weighting choices and prints
the diagnostics that link moment conditions to estimator efficiency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS, IVGMM


PYTHON_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PYTHON_ROOT / "data"


def as_float(value: object) -> float:
    """Convert text-file fields to floats while treating Hansen missing codes as NA."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip()
        if text in {"", ".", "NA"}:
            return float("nan")
        return float(text)
    return float(value)


def load_txt(name: str) -> list[dict[str, float | str]]:
    """Load a tab-delimited dataset into dictionaries for explicit filtering."""
    path = DATA_ROOT / name / f"{name}.txt"
    frame = pd.read_csv(path, sep="\t")
    rows: list[dict[str, float | str]] = []
    for raw_row in frame.to_dict(orient="records"):
        parsed: dict[str, float | str] = {}
        for key, value in raw_row.items():
            try:
                parsed[str(key)] = as_float(value)
            except (TypeError, ValueError):
                parsed[str(key)] = "" if pd.isna(value) else str(value)
        rows.append(parsed)
    return rows


def package_iv_gmm(
    y: np.ndarray,
    exog: pd.DataFrame,
    endog: pd.DataFrame,
    instruments: pd.DataFrame,
    param_names: list[str],
) -> dict[str, object]:
    """Estimate comparable 2SLS and efficient two-step IV-GMM specifications."""
    iv = IV2SLS(y, exog, endog, instruments).fit(cov_type="robust")
    gmm = IVGMM(y, exog, endog, instruments, weight_type="robust").fit(
        iter_limit=2,
        cov_type="robust",
    )
    return {
        "n": int(y.shape[0]),
        "param_names": param_names,
        "beta_2sls": iv.params,
        "se_2sls": iv.std_errors,
        "beta_gmm": gmm.params,
        "se_gmm": gmm.std_errors,
        "j_stat": float(gmm.j_stat.stat),
        "j_df": int(gmm.j_stat.df),
        "j_p": float(gmm.j_stat.pval),
    }


def ajr_results() -> dict[str, object]:
    """Prepare AJR moments using mortality and squared mortality instruments."""
    rows = load_txt("AJR2001")
    required = ["loggdp", "risk", "logmort0"]
    sample = [row for row in rows if all(np.isfinite(float(row[name])) for name in required)]

    loggdp = np.array([float(row["loggdp"]) for row in sample], dtype=float)
    risk = np.array([float(row["risk"]) for row in sample], dtype=float)
    logmort = np.array([float(row["logmort0"]) for row in sample], dtype=float)

    exog = pd.DataFrame({"const": np.ones(len(sample))})
    endog = pd.DataFrame({"risk": risk})
    instruments = pd.DataFrame({"logmort": logmort, "logmort_sq": logmort**2})
    return package_iv_gmm(loggdp, exog, endog, instruments, ["const", "risk"])


def card_sample(instrument_case: str) -> dict[str, object]:
    """Prepare Card wage equations with either baseline or expanded instruments."""
    rows = load_txt("Card1995")
    base_required = ["lwage76", "ed76", "age76", "black", "reg76r", "smsa76r"]
    if instrument_case == "a":
        extra_required = ["nearc4a", "nearc4b"]
    elif instrument_case == "d":
        extra_required = ["nearc4a", "nearc4b"]
    else:
        raise ValueError(f"Unknown case: {instrument_case}")

    sample = [row for row in rows if all(np.isfinite(float(row[name])) for name in base_required + extra_required)]

    lwage = np.array([float(row["lwage76"]) for row in sample], dtype=float)
    ed76 = np.array([float(row["ed76"]) for row in sample], dtype=float)
    age76 = np.array([float(row["age76"]) for row in sample], dtype=float)
    black = np.array([float(row["black"]) for row in sample], dtype=float)
    reg76r = np.array([float(row["reg76r"]) for row in sample], dtype=float)
    smsa76r = np.array([float(row["smsa76r"]) for row in sample], dtype=float)
    nearc4a = np.array([float(row["nearc4a"]) for row in sample], dtype=float)
    nearc4b = np.array([float(row["nearc4b"]) for row in sample], dtype=float)

    exp = age76 - ed76 - 6.0
    exp2 = exp**2 / 100.0

    exog = pd.DataFrame(
        {
            "const": np.ones(len(sample)),
            "exp": exp,
            "exp2": exp2,
            "black": black,
            "reg76r": reg76r,
            "smsa76r": smsa76r,
        }
    )
    endog = pd.DataFrame({"ed76": ed76})

    if instrument_case == "a":
        instruments = pd.DataFrame({"nearc4a": nearc4a, "nearc4b": nearc4b})
    else:
        # Interactions expand the moment set and make the overidentification test meaningful.
        instruments = pd.DataFrame(
            {
                "nearc4a": nearc4a,
                "nearc4b": nearc4b,
                "nearc4a_age76": nearc4a * age76,
                "nearc4a_age2": nearc4a * age76**2 / 100.0,
            }
        )

    return package_iv_gmm(
        lwage,
        exog,
        endog,
        instruments,
        ["const", "ed76", "exp", "exp2", "black", "reg76r", "smsa76r"],
    )


def print_result(label: str, result: dict[str, object]) -> None:
    """Print point estimates, robust SEs, and the GMM overidentification test."""
    print(label)
    print(f"n = {result['n']}")
    print("2SLS estimates and robust SEs")
    for name in result["param_names"]:
        beta = result["beta_2sls"][name]
        se = result["se_2sls"][name]
        print(f"  {name:8s} {float(beta): .9f}  ({float(se):.9f})")
    print("two-step efficient GMM estimates and SEs")
    for name in result["param_names"]:
        beta = result["beta_gmm"][name]
        se = result["se_gmm"][name]
        print(f"  {name:8s} {float(beta): .9f}  ({float(se):.9f})")
    print(
        f"J statistic = {float(result['j_stat']):.9f}, df = {int(result['j_df'])}, "
        f"p-value = {float(result['j_p']):.9f}"
    )
    print()


def main() -> None:
    print_result("Exercise 13.27 (AJR2001)", ajr_results())
    print_result("Exercise 13.28(a) (Card1995)", card_sample("a"))
    print_result("Exercise 13.28(b) / 12.24(d) (Card1995)", card_sample("d"))


if __name__ == "__main__":
    main()
