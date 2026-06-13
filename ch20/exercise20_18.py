"""Replicate Exercise 20.18 with a nonparametric IV classroom-size design.

The script reuses the Angrist-Lavy first-stage structure and plots fitted
effects for math scores, making the instrument-driven prediction explicit.
"""

from __future__ import annotations

import numpy as np

from common import al1999_sample, fit_iv_al, print_title, save_line_plot


def fitted_curve(result, mean_enrollment: float, mean_grade4: float, mean_d: float, mean_c: float):
    """Build prediction functions that vary one covariate while holding others fixed."""
    def class_curve(values: np.ndarray) -> np.ndarray:
        """Vary class size along a grid using the fitted IV coefficients."""
        out = []
        for class_size in values:
            c = class_size / 40.0
            value = (
                result.params["const"]
                + result.params["d"] * mean_d
                + result.params["d2"] * mean_d**2
                + result.params["d3"] * mean_d**3
                + result.params["enrollment"] * mean_enrollment
                + result.params["grade4"] * mean_grade4
                + result.params["c"] * c
                + result.params["c2"] * c**2
                + result.params["c3"] * c**3
                + result.params["cd"] * c * mean_d
            )
            out.append(float(value))
        return np.array(out)

    def disadvantaged_curve(values: np.ndarray) -> np.ndarray:
        """Vary disadvantaged share along a grid using the fitted IV coefficients."""
        out = []
        for disadvantaged in values:
            d = disadvantaged / 14.0
            value = (
                result.params["const"]
                + result.params["d"] * d
                + result.params["d2"] * d**2
                + result.params["d3"] * d**3
                + result.params["enrollment"] * mean_enrollment
                + result.params["grade4"] * mean_grade4
                + result.params["c"] * mean_c
                + result.params["c2"] * mean_c**2
                + result.params["c3"] * mean_c**3
                + result.params["cd"] * mean_c * d
            )
            out.append(float(value))
        return np.array(out)

    return class_curve, disadvantaged_curve


def main() -> None:
    al = al1999_sample()
    print_title("Exercise 20.18")
    print(f"Sample size: {len(al)}")
    print(f"School clusters: {al['schlcode'].nunique()}")

    mean_enrollment = float(al["enrollment"].mean())
    mean_grade4 = float(al["grade4"].mean())
    mean_d = float(al["d"].mean())
    mean_c = float(al["c"].mean())

    for depvar in ["avgverb", "avgmath"]:
        # Fit the same IV specification for reading and math to compare responses.
        _, result = fit_iv_al(depvar)
        print(f"\nDependent variable: {depvar}")
        print(result.params.round(6))
        test = result.wald_test(formula="c = 0, c2 = 0, c3 = 0, cd = 0")
        print(f"Wald statistic for no class-size effect: {float(test.stat):.4f}")
        print(f"Wald p-value: {float(test.pval):.4g}")
        for class_size in [20, 25, 30, 35, 40]:
            class_curve, disadvantaged_curve = fitted_curve(result, mean_enrollment, mean_grade4, mean_d, mean_c)
            print(f"class={class_size:>2}: fitted score={class_curve(np.array([class_size]))[0]:.4f}")
        for disadvantaged in [0, 10, 20, 30, 40, 50]:
            class_curve, disadvantaged_curve = fitted_curve(result, mean_enrollment, mean_grade4, mean_d, mean_c)
            print(f"disadvantaged={disadvantaged:>2}: fitted score={disadvantaged_curve(np.array([disadvantaged]))[0]:.4f}")

    _, read_result = fit_iv_al("avgverb")
    _, math_result = fit_iv_al("avgmath")
    read_class_curve, read_dis_curve = fitted_curve(read_result, mean_enrollment, mean_grade4, mean_d, mean_c)
    math_class_curve, math_dis_curve = fitted_curve(math_result, mean_enrollment, mean_grade4, mean_d, mean_c)

    class_grid = np.linspace(20.0, 40.0, 200)
    dis_grid = np.linspace(0.0, 50.0, 200)

    path = save_line_plot(
        "ch20_ex20_18_classsize.png",
        class_grid,
        [("Reading", read_class_curve(class_grid)), ("Math", math_class_curve(class_grid))],
        xlabel="Class size",
        ylabel="Predicted score",
        title="Exercise 20.18: effect of class size",
    )
    print(f"\nSaved class-size comparison plot to {path}")

    path = save_line_plot(
        "ch20_ex20_18_disadvantaged.png",
        dis_grid,
        [("Reading", read_dis_curve(dis_grid)), ("Math", math_dis_curve(dis_grid))],
        xlabel="Percent disadvantaged",
        ylabel="Predicted score",
        title="Exercise 20.18: effect of disadvantage",
    )
    print(f"Saved disadvantaged comparison plot to {path}")

    print(f"Reading predicted effect of class size 20 -> 40: {read_class_curve(np.array([40]))[0] - read_class_curve(np.array([20]))[0]:.4f}")
    print(f"Math predicted effect of class size 20 -> 40: {math_class_curve(np.array([40]))[0] - math_class_curve(np.array([20]))[0]:.4f}")
    print(f"Reading predicted effect of disadvantaged 0 -> 50: {read_dis_curve(np.array([50]))[0] - read_dis_curve(np.array([0]))[0]:.4f}")
    print(f"Math predicted effect of disadvantaged 0 -> 50: {math_dis_curve(np.array([50]))[0] - math_dis_curve(np.array([0]))[0]:.4f}")


if __name__ == "__main__":
    main()
