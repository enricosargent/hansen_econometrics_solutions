"""Replicate Exercises 24.13-24.14 with wage quantile regressions by sex.

The script estimates conditional quantiles for men and women and reports how
education and experience effects vary across the wage distribution.
"""

from __future__ import annotations

import math

import numpy as np

from qr_tools import fit_quantiles, load_xlsx, log_wage, print_table


TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]


def sample_by_sex(rows: list[dict[str, float | str]], female: int) -> tuple[np.ndarray, np.ndarray]:
    """Select Hispanic workers by sex and return log wages with education."""
    y_values: list[float] = []
    education_values: list[float] = []
    for row in rows:
        lwage = log_wage(row)
        education = float(row["education"])
        if (
            int(row["hisp"]) == 1
            and int(row["female"]) == female
            and education >= 11
            and math.isfinite(lwage)
        ):
            y_values.append(lwage)
            education_values.append(education)
    return np.asarray(y_values), np.asarray(education_values)


def report(rows: list[dict[str, float | str]], female: int, exercise: str, label: str) -> None:
    """Fit and print education-only quantile regressions for one sex."""
    y, education = sample_by_sex(rows, female)
    x = np.column_stack([np.ones(y.size), education])
    estimates = fit_quantiles(y, x, TAUS)
    table_rows = [
        [tau, beta[0], beta[1], beta[0] + 12.0 * beta[1], beta[0] + 16.0 * beta[1]]
        for tau, beta in zip(TAUS, estimates)
    ]
    print(f"{exercise}: {label}")
    print(f"n = {y.size}")
    print_table(["tau", "constant", "education", "q(edu=12)", "q(edu=16)"], table_rows)
    print()


def main() -> None:
    # Loading once keeps the male and female samples directly comparable.
    rows = load_xlsx("cps09mar")
    report(rows, 0, "Exercise 24.13", "Hispanic men with education >= 11")
    report(rows, 1, "Exercise 24.14", "Hispanic women with education >= 11")


if __name__ == "__main__":
    main()
