"""Convert longley/freeny diagnostics and ridge::linearRidge defaults."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor


def automatic_ridge(features, response):
    """Translate linearRidge's automatic corrForm path for full rank."""
    values = np.asarray(features, dtype=float)
    target = np.asarray(response, dtype=float)
    mean_x, mean_y = values.mean(axis=0), target.mean()
    centered = values - mean_x
    scales = np.linalg.norm(centered, axis=0)
    scaled = centered / scales
    target = target - mean_y
    _, singular, right_t = np.linalg.svd(scaled, full_matrices=False)
    eigenvalues = singular**2
    right = right_t.T
    scores = scaled @ right
    coefficients = scores.T @ target / eigenvalues
    fraction = np.cumsum(eigenvalues) / eigenvalues.sum()
    maximum = int(np.searchsorted(fraction, 0.9) + 1)
    candidates, distance = [], []
    for count in range(1, maximum + 1):
        residual = target - scores[:, :count] @ coefficients[:count]
        variance = residual @ residual / (len(target) - count)
        penalty = count*variance / \
            (coefficients[:count] @ coefficients[:count])
        effective = np.sum(eigenvalues**2 / (eigenvalues + penalty)**2)
        candidates.append(penalty)
        distance.append(abs(effective - count))
    selected = int(np.argmin(distance))
    penalty = candidates[selected]
    shrinkage = coefficients*eigenvalues / (eigenvalues + penalty)
    slopes_scaled = right @ shrinkage
    reference = Ridge(alpha=penalty, fit_intercept=False, solver="svd")
    reference.fit(scaled, target)
    assert np.allclose(reference.coef_, slopes_scaled, rtol=1e-10, atol=1e-9)
    slopes = slopes_scaled / scales
    intercept = mean_y - mean_x @ slopes
    return penalty, selected + 1, np.r_[intercept, slopes]


def report(name, frame, response, predictors):
    target = frame[response].to_numpy()
    features = frame[predictors]
    design = sm.add_constant(features)
    fit = sm.OLS(target, design).fit()
    vifs = np.array([variance_inflation_factor(design.to_numpy(), j)
                     for j in range(1, design.shape[1])])
    manual = []
    for column in predictors:
        remaining = features.drop(columns=column)
        partial = sm.OLS(features[column], sm.add_constant(remaining)).fit()
        manual.append(1 / (1 - partial.rsquared))
    assert np.allclose(vifs, manual, rtol=1e-8)
    centered = features.to_numpy() - features.to_numpy().mean(axis=0)
    scaled = centered / np.linalg.norm(centered, axis=0)
    condition = np.linalg.cond(scaled)
    penalty, components, coefficient = automatic_ridge(features, target)
    print(name, "response:", response, "rows:", len(frame))
    print("Predictors:", ", ".join(predictors))
    print("VIF:", np.array2string(vifs, precision=6))
    print(f"Centered/unit-norm condition X={condition:.6f}, "
          f"X'X={condition**2:.6f}")
    print(f"OLS training R-squared={fit.rsquared:.9f}")
    print(f"Ridge automatic penalty={penalty:.12g}, "
          f"selected PC count={components}")
    print("Ridge raw-scale coefficients:",
          np.array2string(coefficient, precision=6))
    print("SVD ridge / sklearn fixed-penalty agreement: passed")
    return vifs


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    longley_path = data_dir / "ch14-longley.csv"
    longley = pd.read_csv(longley_path).drop(columns="rownames")
    predictors = [c for c in longley if c != "GNP.deflator"]
    report("longley", longley, "GNP.deflator", predictors)
    reduced = ["Unemployed", "Armed.Forces", "Employed"]
    report("longley reduced", longley, "GNP.deflator", reduced)
    freeny = pd.read_csv(data_dir / "ch14-freeny.csv").drop(columns="rownames")
    report("freeny", freeny, "y", [c for c in freeny if c != "y"])
    x = np.array([2., 5., 10., 12.])
    exact = np.column_stack([x, 2*x])
    near = np.column_stack([x, [4., 9., 19., 25.]])
    print("Source toy ranks (without intercept):",
          np.linalg.matrix_rank(exact), np.linalg.matrix_rank(near))
    assert np.linalg.matrix_rank(exact) == 1
    assert np.linalg.matrix_rank(near) == 2


if __name__ == "__main__":
    main()
