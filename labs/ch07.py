"""Simple regression: exact checks and a Python port of the insurance workflow.

The default data are an original construction, not the course CSV.
If a CSV path is supplied, it must contain numeric columns age and male.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def fit_and_check(age, male):
    """Fit an intercept model and verify the centered formulas."""
    x = np.asarray(age, dtype=float)
    y = np.asarray(male, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) <= 2:
        raise ValueError("Need more than two paired, one-dimensional rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(
            "Missing/nonfinite rows require an explicit data policy")
    xc = x - x.mean()
    yc = y - y.mean()
    sxx = xc @ xc
    if sxx <= 0:
        raise ValueError("A constant predictor cannot identify a slope")
    slope = (xc @ yc) / sxx
    intercept = y.mean() - slope*x.mean()
    design = sm.add_constant(x, has_constant="add")
    fit = sm.OLS(y, design, missing="raise").fit()
    np.testing.assert_allclose(fit.params, [intercept, slope], atol=1e-12)
    np.testing.assert_allclose(design.T @ fit.resid, 0, atol=1e-10)
    if yc @ yc > 0:
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = fit.rsquared
        np.testing.assert_allclose(r_squared, correlation**2, atol=1e-12)
    else:
        correlation, r_squared = np.nan, np.nan
    variance = (fit.resid @ fit.resid)/(len(x) - 2)
    np.testing.assert_allclose(fit.bse[1]**2, variance/sxx, atol=1e-12)
    pred = fit.get_prediction([[1.0, x.mean()]]).summary_frame(alpha=0.05)
    assert pred.obs_ci_upper.iloc[0] >= pred.mean_ci_upper.iloc[0]
    print("intercept slope:", np.round(fit.params, 9))
    print("Sxx SSE residual_variance:",
          np.round([sxx, fit.ssr, variance], 9))
    print("correlation R_squared:", np.round([correlation, r_squared], 9))
    print("slope_se t_statistic p_value:",
          np.round([fit.bse[1], fit.tvalues[1], fit.pvalues[1]], 9))
    print("95pct slope_interval:", np.round(fit.conf_int()[1], 9))
    print("at_x_mean: mean_interval prediction_interval:",
          np.round(pred[["mean_ci_lower", "mean_ci_upper", "obs_ci_lower",
                         "obs_ci_upper"]].to_numpy()[0], 9))
    return fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Course data with columns age and male")
    args = parser.parse_args()
    np.set_printoptions(precision=9, suppress=True)
    if args.csv:
        table = pd.read_csv(args.csv)
        print("data: supplied CSV; not authenticated automatically")
        fit_and_check(table["age"], table["male"])
    else:
        print("data: original teaching construction; "
              "not course insurance data")
        x = np.arange(1.0, 7.0)
        noise = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0])
        y = 2.0 + 3.0*x + noise
        fit = fit_and_check(x, y)
        np.testing.assert_allclose(fit.params, [2.0, 3.0], atol=1e-12)
        order = np.array([5, 0, 3, 1, 4, 2])
        permuted = np.linalg.lstsq(
            np.column_stack([np.ones(6), x[order]]), y[order], rcond=None)[0]
        np.testing.assert_allclose(permuted, fit.params, atol=1e-12)
        reversed_slope = np.linalg.lstsq(
            np.column_stack([np.ones(6), -x]), y, rcond=None)[0][1]
        print("slope_after_x_sign_reversal:", round(reversed_slope, 9))
        no_intercept = sm.OLS(y, x[:, None]).fit()
        print("no_intercept_slope residual_sum:",
              np.round([no_intercept.params[0], no_intercept.resid.sum()], 9))
        print("critical_t_df4:", round(stats.t.ppf(0.975, 4), 9))
        intervals = fit.get_prediction([[1.0, 3.5], [1.0, 8.0]])
        table = intervals.summary_frame(alpha=0.05)
        half_widths = (table.mean_ci_upper - table.mean_ci_lower)/2
        np.testing.assert_allclose(
            half_widths, [1.133478968, 3.194492272], atol=1e-9)
        print("mean_interval_half_widths_at_3.5_and_8:",
              np.round(half_widths.to_numpy(), 9))
        print("extrapolated_mean_and_mean_interval_at_8:",
              np.round(table.iloc[1][[
                  "mean", "mean_ci_lower", "mean_ci_upper"]].to_numpy(), 9))
        print("all assertions passed")


if __name__ == "__main__":
    main()
