"""Original AR1 regression construction implementing the lecture's methods."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


def prais_winsten(y, design, rho):
    """Known-rho stationary whitening, including the first observation."""
    if not abs(rho) < 1:
        raise ValueError("Stationary whitening requires abs(rho)<1")
    transformed_y = np.r_[np.sqrt(1-rho**2)*y[0], y[1:]-rho*y[:-1]]
    transformed_x = np.vstack([
        np.sqrt(1-rho**2)*design[0], design[1:]-rho*design[:-1]])
    return sm.OLS(transformed_y, transformed_x).fit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots", help="Output folder for residual-order figure")
    args = parser.parse_args()
    np.set_printoptions(precision=9, suppress=True)
    n, rho, innovation_sd = 80, 0.7, 0.5
    rng = np.random.default_rng(20261103)
    row = np.arange(n)
    x = np.sin(0.35*row) + np.linspace(-1, 1, n)
    design = sm.add_constant(x)
    error = np.empty(n)
    error[0] = rng.normal(scale=innovation_sd/np.sqrt(1-rho**2))
    for i in range(1, n):
        error[i] = rho*error[i-1]+rng.normal(scale=innovation_sd)
    y = design @ np.array([2.0, 3.0]) + error
    covariance = innovation_sd**2/(1-rho**2)*rho**np.abs(row[:, None]-row)
    ols = sm.OLS(y, design).fit()
    gls = sm.GLS(y, design, sigma=covariance).fit()
    pw = prais_winsten(y, design, rho)
    np.testing.assert_allclose(gls.params, pw.params, atol=1e-11)
    whitening = np.eye(n)
    whitening[0, 0] = np.sqrt(1-rho**2)
    whitening[np.arange(1, n), np.arange(n-1)] = -rho
    np.testing.assert_allclose(whitening @ covariance @ whitening.T,
                               innovation_sd**2*np.eye(n), atol=1e-12)
    a = np.linalg.pinv(design)
    true_ols_cov = a @ covariance @ a.T
    true_gls_cov = np.linalg.inv(
        design.T @ np.linalg.solve(covariance, design))
    assert np.linalg.eigvalsh(true_ols_cov-true_gls_cov).min() > -1e-12
    residual = ols.resid
    dw = np.sum(np.diff(residual)**2)/np.sum(residual**2)
    lag_ratio = residual[1:] @ residual[:-1]/(residual @ residual)
    boundary = (residual[0]**2+residual[-1]**2)/(residual @ residual)
    np.testing.assert_allclose(dw, 2*(1-lag_ratio)-boundary, atol=1e-12)
    np.testing.assert_allclose(dw, durbin_watson(residual), atol=1e-12)
    co = sm.OLS(y[1:]-rho*y[:-1], design[1:]-rho*design[:-1]).fit()
    fitted = ols
    converged = False
    for iteration in range(1, 101):
        e = y-design @ fitted.params
        estimated_rho = float(np.clip((e[:-1] @ e[1:])/(e[:-1] @ e[:-1]),
                                      -0.98, 0.98))
        newfit = prais_winsten(y, design, estimated_rho)
        if np.max(np.abs(newfit.params-fitted.params)) < 1e-10:
            fitted = newfit
            converged = True
            break
        fitted = newfit
    augmented = np.column_stack([np.ones(n-1), y[:-1], x[1:], x[:-1]])
    assert np.linalg.matrix_rank(augmented) == 4
    initial = sm.OLS(y[1:], augmented).fit().params[1]
    durbin_fit = prais_winsten(y, design, float(np.clip(initial, -0.98, 0.98)))
    print("data: original seeded AR1 construction, not source observations")
    print("n rho innovation_sd:", n, rho, innovation_sd)
    print("OLS coefficients:", np.round(ols.params, 9))
    print("known_covariance_GLS coefficients:", np.round(gls.params, 9))
    print("Prais_Winsten agrees with GLS: passed")
    print("conditional_Cochrane_Orcutt coefficients:", np.round(co.params, 9))
    print("true_OLS_standard_deviations:", np.round(
        np.sqrt(np.diag(true_ols_cov)), 9))
    print("true_GLS_standard_deviations:", np.round(
        np.sqrt(np.diag(true_gls_cov)), 9))
    print("DW lag_ratio endpoint_correction:",
          np.round([dw, lag_ratio, boundary], 9))
    null_rng = np.random.default_rng(20261104)
    null_error = null_rng.normal(size=(n, 9999))
    null_residual = null_error-design @ (a @ null_error)
    null_numerator = np.sum(np.diff(null_residual, axis=0)**2, axis=0)
    null_dw = null_numerator/np.sum(null_residual**2, axis=0)
    count = int(np.count_nonzero(null_dw <= dw))
    print("Gaussian_fixed_design_DW_MC: lower_tail_count draws p_plus1:",
          count, len(null_dw), (count+1)/(len(null_dw)+1))
    print("iterated_PW rho iterations converged:",
          round(estimated_rho, 9), iteration, converged)
    print("iterated_PW coefficients:", np.round(fitted.params, 9))
    print("Durbin_initial_rho:", round(initial, 9))
    print("Durbin_initialized_PW coefficients:",
          np.round(durbin_fit.params, 9))
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        target = Path(args.plots)
        target.mkdir(parents=True, exist_ok=True)
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(row+1, residual, "o-", markersize=3)
            axes[0].axhline(0, color="gray")
            axes[0].set(xlabel="Observation in time order",
                        ylabel="OLS residual")
            axes[1].scatter(residual[:-1], residual[1:], s=12)
            axes[1].set(xlabel="Previous residual", ylabel="Next residual")
            fig.tight_layout()
            fig.savefig(target / "ch12-ar1-residuals.svg")
            plt.close(fig)
    # Optional reference CSV is deterministic and retained inside the project.
    target = Path(__file__).resolve().parent / "data/ch12-ar1-construction.csv"
    pd.DataFrame({"x": x, "y": y}).to_csv(target, index=False)
    print("all assertions passed: whitening, GLS, "
          "covariance order, DW identity")


if __name__ == "__main__":
    main()
