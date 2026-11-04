"""Convert the lecture's ToothGrowth normality example to Python."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

matplotlib.use("Agg")


def qq_coordinates(values):
    """R qqnorm ppoints, qqline type-7 quartiles, car default envelope."""
    ordered = np.sort(np.asarray(values, dtype=float))
    n = ordered.size
    offset = 0.375 if n <= 10 else 0.5
    probabilities = (np.arange(1, n + 1) - offset) / (n + 1 - 2*offset)
    theoretical = stats.norm.ppf(probabilities)
    quartiles = np.quantile(ordered, [0.25, 0.75], method="linear")
    normal_quartiles = stats.norm.ppf([0.25, 0.75])
    slope = np.diff(quartiles)[0] / np.diff(normal_quartiles)[0]
    intercept = quartiles[0] - slope*normal_quartiles[0]
    fitted = intercept + slope*theoretical
    se = slope / stats.norm.pdf(theoretical)
    se *= np.sqrt(probabilities*(1 - probabilities) / n)
    halfwidth = stats.norm.ppf(0.975)*se
    return ordered, theoretical, fitted, halfwidth, intercept, slope


def main():
    here = Path(__file__).resolve().parent
    data = pd.read_csv(here / "data/ch13-toothgrowth.csv")
    length = data["len"].to_numpy()
    assert len(length) == 60 and not np.isnan(length).any()
    ordered, theoretical, fitted, halfwidth, a, b = qq_coordinates(length)
    result = stats.shapiro(length)
    shifted = stats.shapiro(3 + 2*length)
    assert np.isclose(result.statistic, shifted.statistic, atol=1e-12)
    print(f"ToothGrowth rows: {len(length)}")
    print(f"Shapiro-Wilk W={result.statistic:.9f}, p={result.pvalue:.9f}")
    print(f"Quartile reference line: intercept={a:.6f}, slope={b:.6f}")
    inside = np.abs(ordered - fitted) <= halfwidth
    print(f"Inside approximate pointwise 95% envelope: {inside.sum()}/60")
    print("Affine-invariance check: passed")
    design = np.column_stack([
        np.ones(len(data)), (data["supp"] == "VC").astype(float), data["dose"]
    ])
    coefficient = np.linalg.lstsq(design, length, rcond=None)[0]
    residual = length - design @ coefficient
    residual_result = stats.shapiro(residual)
    print("Model: len ~ intercept + I(supp == VC) + numeric dose")
    print("Coefficients:", np.array2string(coefficient, precision=6))
    print(f"Residual Shapiro W={residual_result.statistic:.9f}, "
          f"nominal iid p={residual_result.pvalue:.9f}")
    print("Residual p-value is descriptive, not exact regression calibration.")
    sample = np.array([-1.0, 0.0, 1.0])
    distribution = stats.norm.cdf(sample)
    right = np.arange(1, 4) / 3 - distribution
    left = distribution - np.arange(3) / 3
    discrepancy = max(right.max(), left.max())
    print(f"Three-point standard-normal KS distance={discrepancy:.9f}")
    assert np.isclose(discrepancy, stats.kstest(sample, "norm").statistic)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(length, bins="sturges", edgecolor="white")
    axes[0].set(xlabel="Odontoblast length", ylabel="Count")
    axes[0].set_title("ToothGrowth response; Sturges bin count")
    axes[1].fill_between(theoretical, fitted - halfwidth,
                         fitted + halfwidth, alpha=0.2)
    axes[1].plot(theoretical, fitted, color="firebrick")
    axes[1].scatter(theoretical, ordered, s=14)
    axes[1].set(xlabel="Standard-normal quantile", ylabel="Observed length")
    axes[1].set_title("Quartile line and pointwise 95% envelope")
    figure.tight_layout()
    figure.savefig(here / "ch13-normality.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
