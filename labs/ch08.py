"""Multiple regression checks and ports of source car/housing workflows.

Default: original deterministic construction. Optional CSV paths execute the
source methods on supplied authentic files without inventing data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def construction():
    x1 = np.arange(-3.5, 4.0)
    x2 = 0.6*x1 + np.array([-1., 1., -1., 1., 1., -1., 1., -1.])
    noise = np.array([1., 1., -1., -1., -1., -1., 1., 1.])
    design = np.column_stack([np.ones(8), x1, x2])
    y = design @ np.array([10., -2., 3.]) + noise
    fit = sm.OLS(y, design).fit()
    simple = sm.OLS(y, sm.add_constant(x1)).fit()
    q = design.shape[1]
    projection = design @ np.linalg.pinv(design)
    complement = np.eye(len(y)) - projection
    np.testing.assert_allclose(fit.params, [10., -2., 3.], atol=1e-12)
    np.testing.assert_allclose(projection @ projection, projection, atol=1e-12)
    np.testing.assert_allclose(complement @ design, 0, atol=1e-12)
    np.testing.assert_allclose(design.T @ fit.resid, 0, atol=1e-12)
    np.testing.assert_allclose(np.trace(projection), q, atol=1e-12)
    independent = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(fit.params, independent, atol=1e-12)
    print("data: original construction, not source Toyota rows")
    print("simple_slope adjusted_slope:",
          np.round([simple.params[1], fit.params[1]], 9))
    print("multiple_coefficients:", np.round(fit.params, 9))
    print("n q residual_df:", len(y), q, int(fit.df_resid))
    print("SSE unbiased_variance mle_variance:",
          np.round([fit.ssr, fit.mse_resid, fit.ssr/len(y)], 9))
    print("projection_trace residual_projection_trace:",
          np.round([np.trace(projection), np.trace(complement)], 9))
    print("partial_F p_value df_difference:",
          np.round(fit.compare_f_test(simple), 9))


def toyota(path, training_rows=None):
    data = pd.read_csv(path)
    data = data.rename(columns={"Age_08_04": "Age"})
    if training_rows is None:
        raise ValueError("Supply exact zero-based source training row indices")
    indices = np.asarray(training_rows, dtype=int)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Training rows must be unique")
    train = data.iloc[indices]
    validation = data.drop(data.index[indices])
    simple = smf.ols("Price ~ KM", data=train, missing="raise").fit()
    adjusted = smf.ols("Price ~ KM + Age", data=train, missing="raise").fit()
    predictors = [c for c in data.columns if c != "Price"]
    all_formula = "Price ~ " + " + ".join(f"Q({c!r})" for c in predictors)
    complete = smf.ols(all_formula, data=train, missing="raise").fit()
    print("Toyota training validation rows:", len(train), len(validation))
    print("Toyota simple adjusted KM:", simple.params["KM"],
          adjusted.params["KM"])
    print("Toyota all predictor coefficients:", complete.params.to_dict())
    print("Toyota validation predictions:",
          complete.predict(validation).to_numpy())


def housing(path, new_path=None, plots=None):
    data = pd.read_csv(path).rename(columns={
        "Total.Units": "Units", "Gross.SqFt": "SqFt",
        "Market.Value.per.SqFt": "ValuePerSqFt"})
    formula = "ValuePerSqFt ~ Units + SqFt + C(Boro)"
    fit = smf.ols(formula, data=data, missing="raise").fit()
    print("housing source rows residual_df:", int(fit.nobs), int(fit.df_resid))
    print("housing coefficients:")
    for name, coefficient in fit.params.items():
        print(name, f"{coefficient:.9f}")
    print("housing R_squared:", f"{fit.rsquared:.9f}")
    print("housing coefficient_intervals:")
    print(np.round(fit.conf_int().to_numpy(), 9))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        target = Path(plots)
        target.mkdir(parents=True, exist_ok=True)
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            categories = sorted(data["Boro"].unique())
            groups = [data.loc[data.Boro == b, "ValuePerSqFt"]
                      for b in categories]
            axes[0].boxplot(groups, tick_labels=categories)
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].set_ylabel("Market value per square foot")
            axes[1].hist(groups, bins=25, label=categories, histtype="step")
            axes[1].set_xlabel("Market value per square foot")
            axes[1].set_ylabel("Buildings")
            axes[1].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(target / "ch08-housing-distributions.svg")
            plt.close(fig)
            fig, axes = plt.subplots(len(categories), 1, figsize=(7, 9))
            for axis, category, values in zip(axes, categories, groups):
                axis.hist(values, bins=25)
                axis.set_title(category, fontsize=9)
                axis.set_ylabel("Buildings", fontsize=8)
            axes[-1].set_xlabel("Market value per square foot")
            fig.tight_layout()
            fig.savefig(target / "ch08-housing-facets.svg")
            plt.close(fig)
            fig, axis = plt.subplots(figsize=(9, 5))
            coefficients = fit.params.iloc[:-1]
            intervals = fit.conf_int().iloc[:-1].to_numpy()
            error = np.vstack([coefficients.to_numpy() - intervals[:, 0],
                               intervals[:, 1] - coefficients.to_numpy()])
            axis.errorbar(
                coefficients.to_numpy(), np.arange(len(coefficients)),
                xerr=error, fmt="o", capsize=3)
            axis.set_yticks(np.arange(len(coefficients)), coefficients.index)
            axis.axvline(0, color="gray", linewidth=1)
            axis.set_xlabel(
                "Coefficient and 95% normal-theory confidence interval")
            axis.set_title("SqFt omitted from this scale; see numeric table")
            fig.tight_layout()
            fig.savefig(target / "ch08-housing-coefficients.svg")
            plt.close(fig)
    if new_path:
        new = pd.read_csv(new_path)
        predictions = fit.get_prediction(new).summary_frame(alpha=0.05)
        assert np.all(predictions.obs_ci_lower < predictions.mean_ci_lower)
        assert np.all(predictions.obs_ci_upper > predictions.mean_ci_upper)
        print("housing new rows:", len(new))
        print("housing first2 predictions: mean mean_se mean_CI obs_CI")
        print(np.round(predictions.iloc[:2].to_numpy(), 9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--housing")
    parser.add_argument("--new")
    parser.add_argument("--toyota")
    parser.add_argument("--training-rows", nargs="+", type=int)
    parser.add_argument(
        "--plots", help="Folder for converted housing graphics")
    args = parser.parse_args()
    np.set_printoptions(precision=9, suppress=True)
    construction()
    if args.housing:
        housing(args.housing, args.new, args.plots)
    if args.toyota:
        toyota(args.toyota, args.training_rows)
    print("all executed assertions passed")


if __name__ == "__main__":
    main()
