"""Chapter 1: exact source correlation examples and original checks."""

import numpy as np


def main():
    x = np.array([1.0, 3.0, 9.0, 11.0])
    y = np.array([5.0, 9.0, 21.0, 25.0])
    centered = np.column_stack((x - x.mean(), y - y.mean()))
    population = centered.T @ centered / len(x)
    sample = centered.T @ centered / (len(x) - 1)
    assert np.allclose(population, [[17, 34], [34, 68]])
    assert np.allclose(sample, population * 4 / 3)
    print("Source linear pairs: x =", x.astype(int).tolist())
    print("Source linear pairs: y =", y.astype(int).tolist())
    print(f"Means: x={x.mean():.6f}, y={y.mean():.6f}")
    print(f"Covariance, divisor n: {population[0, 1]:.6f}")
    print(f"Covariance, divisor n-1: {sample[0, 1]:.6f}")
    print(
        f"Variances, divisor n: {population[0, 0]:.6f}, {population[1, 1]:.6f}"
    )
    print(f"Pearson correlation: {np.corrcoef(x, y)[0, 1]:.6f}")
    transformed = np.corrcoef(10 + 100 * x, 7 - 2 * y)[0, 1]
    assert np.isclose(transformed, -1)
    print(f"Correlation after positive/negative rescaling: {transformed:.6f}")
    xq = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    yq = xq**2
    covariance = np.mean((xq - xq.mean()) * (yq - yq.mean()))
    assert abs(covariance) < 1e-12
    assert np.allclose(yq, xq**2)
    print(f"Source quadratic mean response: {yq.mean():.6f}")
    print(f"Source quadratic covariance: {covariance:.6f}")
    print(f"Source quadratic correlation: {np.corrcoef(xq, yq)[0, 1]:.6f}")
    print(f"Quadratic reconstruction error: {np.mean((yq - xq**2) ** 2):.6f}")
    print(
        f"Constant-predictor mean squared error: "
        f"{np.mean((yq - yq.mean()) ** 2):.6f}"
    )
    codes_a = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    codes_b = np.array([0.0, 0.0, 2.0, 2.0, 1.0, 1.0])
    response = np.array([0.0, 1.0, 3.0, 4.0, 6.0, 7.0])
    print(
        f"Arbitrary category coding correlation A: "
        f"{np.corrcoef(codes_a, response)[0, 1]:.6f}"
    )
    print(
        f"Arbitrary category coding correlation B: "
        f"{np.corrcoef(codes_b, response)[0, 1]:.6f}"
    )
    print("Checks passed: source arithmetic, scaling, nonlinear dependence.")


if __name__ == "__main__":
    main()
