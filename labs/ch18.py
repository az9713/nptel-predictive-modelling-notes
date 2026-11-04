"""Synthetic sharp regression-discontinuity mechanisms and assumptions."""
from pathlib import Path

import matplotlib
import numpy as np
import statsmodels.api as sm
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt

matplotlib.use("Agg")


def local_linear(running, outcome, cutoff=0., bandwidth=0.5):
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")
    centered = np.asarray(running) - cutoff
    keep = abs(centered) < bandwidth
    x = centered[keep]
    treatment = (x >= 0).astype(float)
    design = np.column_stack([np.ones(len(x)), treatment, x, treatment*x])
    weights = 1 - abs(x) / bandwidth
    if np.linalg.matrix_rank(design) < 4:
        raise ValueError("Both cutoff sides need enough distinct observations")
    fit = sm.WLS(np.asarray(outcome)[keep], design, weights=weights).fit()
    direct = np.linalg.lstsq(np.sqrt(weights)[:, None]*design,
                             np.sqrt(weights)*np.asarray(outcome)[keep],
                             rcond=None)[0]
    assert np.allclose(fit.params, direct, atol=1e-10)
    robust = fit.get_robustcov_results(cov_type="HC3")
    return fit, robust.bse[1], int((x < 0).sum()), int((x >= 0).sum())


def main():
    rng = np.random.default_rng(1801)
    running = rng.uniform(-1, 1, 1200)
    treatment = (running >= 0).astype(float)
    untreated = 1 + 0.6*running + 0.4*running**2
    effect = 2 + 0.3*running
    disturbance = rng.normal(scale=0.35, size=len(running))
    outcome = untreated + treatment*effect + disturbance
    print("Original synthetic sharp design; seed=1801, n=1200")
    print("True treatment effect at cutoff=2.000000")
    fits = []
    for bandwidth in [0.25, 0.5, 0.75, 1.0]:
        fit, standard_error, left, right = local_linear(
            running, outcome, bandwidth=bandwidth)
        fits.append((bandwidth, fit))
        print(f"h={bandwidth:.2f}, nLeft={left}, nRight={right}, "
              f"jump={fit.params[1]:.9f}, "
              f"conventional HC3SE={standard_error:.9f}")
    print("HC3 values are conventional demo standard errors, "
          "not RD bias-corrected inference.")
    nuisance_jump = 0.8
    invalid = outcome + nuisance_jump*treatment
    valid_fit = local_linear(running, outcome, bandwidth=0.5)[0]
    invalid_fit = local_linear(running, invalid, bandwidth=0.5)[0]
    jump_change = invalid_fit.params[1] - valid_fit.params[1]
    assert np.isclose(jump_change, nuisance_jump)
    print(f"Untreated-mean violation adds={nuisance_jump:.6f} "
          "to observed jump, "
          "without changing treatment effect.")
    smooth = 1 + np.tanh(4*running) + disturbance
    smooth_fit = local_linear(running, smooth, bandwidth=1.)[0]
    narrow = local_linear(running, smooth, bandwidth=0.25)[0]
    print("Continuous nonlinear mean, true jump=0: "
          f"h1fit={smooth_fit.params[1]:.9f}, "
          f"h0.25fit={narrow.params[1]:.9f}")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, response, title in [
        (axes[0],
         outcome, "Sharp design; true jump2"),
        (axes[1],
         smooth, "Continuous nonlinear mean")]:
        means, edges, _ = binned_statistic(running, response, bins=30)
        midpoint = (edges[:-1] + edges[1:]) / 2
        axis.scatter(midpoint, means, s=18, label="Binned observations")
        fitted = local_linear(running, response, bandwidth=0.5)[0]
        for left, right in [(-0.5, 0), (0, 0.5)]:
            grid = np.linspace(left, right, 100)
            indicator = np.full(len(grid), float(left >= 0))
            design = np.column_stack([np.ones(len(grid)), indicator,
                                      grid, indicator*grid])
            axis.plot(grid, fitted.predict(design), color="firebrick")
        axis.axvline(0, color="gray", linestyle="--")
        axis.set(
            xlabel="Running variable relative to cutoff",
            ylabel="Outcome")
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(Path(__file__).with_name("ch18-cutoff.svg"))
    plt.close(figure)
    print("Local WLS/direct-whitening and nuisance-jump assertions: passed")


if __name__ == "__main__":
    main()
