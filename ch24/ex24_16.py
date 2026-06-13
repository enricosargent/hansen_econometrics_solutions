"""Replicate Exercise 24.16 with wage quantile regressions for college women.

The script estimates race-specific conditional quantiles to show how covariate
effects differ across both groups and locations in the wage distribution.
"""

from __future__ import annotations

import math

import numpy as np

from qr_tools import fit_quantiles, load_xlsx, log_wage, polynomial_design, print_table


TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]
GRID = np.array([0.0, 10.0, 20.0, 30.0, 40.0])


def college_women_sample(
    rows: list[dict[str, float | str]],
    race_code: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select college-educated women of one race and return log wage by experience."""
    y_values: list[float] = []
    experience_values: list[float] = []
    for row in rows:
        lwage = log_wage(row)
        if (
            int(row["female"]) == 1
            and int(row["race"]) == race_code
            and float(row["education"]) == 16.0
            and math.isfinite(lwage)
        ):
            experience = float(row["age"]) - float(row["education"]) - 6.0
            y_values.append(lwage)
            experience_values.append(experience)
    return np.asarray(y_values), np.asarray(experience_values)


def report(rows: list[dict[str, float | str]], race_code: int, label: str) -> None:
    """Fit polynomial-experience quantile regressions and report fitted profiles."""
    y, experience = college_women_sample(rows, race_code)
    x = polynomial_design(experience)
    estimates = fit_quantiles(y, x, TAUS)
    coefficient_rows = [[tau, *beta] for tau, beta in zip(TAUS, estimates)]
    grid_x = polynomial_design(GRID)
    # Multiplying by the design grid turns coefficients into conditional quantiles.
    fitted = estimates @ grid_x.T
    fitted_rows = [[tau, *values] for tau, values in zip(TAUS, fitted)]

    print(f"Exercise 24.16: {label}, education = 16")
    print(f"n = {y.size}, experience range = [{experience.min():.0f}, {experience.max():.0f}]")
    print("Polynomial uses z = (experience - 20) / 10.")
    print_table(["tau", "z^0", "z^1", "z^2", "z^3", "z^4", "z^5"], coefficient_rows)
    print("Predicted log wages")
    print_table(["tau", "exp=0", "exp=10", "exp=20", "exp=30", "exp=40"], fitted_rows)
    print()


def main() -> None:
    rows = load_xlsx("cps09mar")
    report(rows, 2, "Black women")
    report(rows, 1, "White women")


if __name__ == "__main__":
    main()
