"""Convert Credit interaction and MASS Boston polynomial examples to Python."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", help="Folder for converted source graphics")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    np.set_printoptions(precision=9, suppress=True)
    credit = pd.read_csv(here / "data/ch10-credit.csv")
    credit["Student"] = pd.Categorical(
        credit.Student, categories=["No", "Yes"])
    additive = smf.ols("Balance ~ Income + C(Student)", data=credit).fit()
    interaction = smf.ols("Balance ~ Income*C(Student)", data=credit).fit()
    expanded = smf.ols(
        "Balance ~ Income + C(Student) + Income:C(Student)", data=credit).fit()
    np.testing.assert_allclose(interaction.fittedvalues, expanded.fittedvalues)
    credit["Income_centered"] = credit.Income-credit.Income.mean()
    centered = smf.ols(
        "Balance ~ Income_centered*C(Student)", data=credit).fit()
    np.testing.assert_allclose(centered.fittedvalues,
                               interaction.fittedvalues, atol=1e-9)
    print("Credit rows:", len(credit))
    print("additive coefficients:", additive.params.to_dict())
    print("interaction coefficients:", interaction.params.to_dict())
    print("additive interaction adjusted_R_squared:",
          np.round([additive.rsquared_adj, interaction.rsquared_adj], 9))
    print("interaction_partial_F p_value df:",
          np.round(interaction.compare_f_test(additive), 9))
    for income in [20., 100.]:
        table = pd.DataFrame(
            {"Income": [income, income], "Student": ["No", "Yes"]})
        prediction = interaction.predict(table).to_numpy()
        print("Income No Yes difference:", income,
              np.round(np.r_[prediction, prediction[1]-prediction[0]], 9))
    boston = pd.read_csv(here / "data/ch11-boston.csv")
    linear = smf.ols("medv ~ lstat", data=boston).fit()
    quadratic = smf.ols("medv ~ lstat + I(lstat**2)", data=boston).fit()
    boston["centered"] = boston.lstat-boston.lstat.mean()
    shifted = smf.ols("medv ~ centered + I(centered**2)", data=boston).fit()
    np.testing.assert_allclose(shifted.fittedvalues,
                               quadratic.fittedvalues, atol=1e-9)
    assert quadratic.ssr < linear.ssr
    print("Boston rows:", len(boston))
    print("linear coefficients:", np.round(linear.params.to_numpy(), 9))
    print("quadratic coefficients:", np.round(quadratic.params.to_numpy(), 9))
    print("linear quadratic SSE:", np.round([linear.ssr, quadratic.ssr], 9))
    f_value, p_value, df_difference = quadratic.compare_f_test(linear)
    print("quadratic_partial_F p_value df:", f"{f_value:.9f}",
          f"{p_value:.12e}", int(df_difference))
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        target = Path(args.plots)
        target.mkdir(parents=True, exist_ok=True)
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            grid = np.linspace(credit.Income.min(), credit.Income.max(), 200)
            titles = ["Additive: shared slope", "Interaction: group slopes"]
            for axis, model, title in zip(
                    axes, [additive, interaction], titles):
                for group in ["No", "Yes"]:
                    subset = credit[credit.Student == group]
                    axis.scatter(subset.Income, subset.Balance, s=6, alpha=0.3)
                    new = pd.DataFrame({"Income": grid, "Student": group})
                    axis.plot(grid, model.predict(new), label=group)
                axis.set(title=title, xlabel="Income, thousands of dollars")
                axis.legend(title="Student")
            axes[0].set_ylabel("Simulated credit balance, dollars")
            fig.tight_layout()
            fig.savefig(target / "ch11-credit-interaction.svg")
            plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            grid = np.linspace(boston.lstat.min(), boston.lstat.max(), 200)
            new = pd.DataFrame({"lstat": grid})
            axes[0].scatter(boston.lstat, boston.medv, s=8, alpha=0.4)
            axes[0].plot(grid, linear.predict(new), label="Linear")
            axes[0].plot(grid, quadratic.predict(new), label="Quadratic")
            axes[0].set(xlabel="lstat, historical percentage score",
                        ylabel="Median value, thousands of dollars")
            axes[0].legend()
            axes[1].scatter(linear.fittedvalues, linear.resid, s=8, alpha=0.4)
            trend = sm.nonparametric.lowess(linear.resid, linear.fittedvalues)
            axes[1].plot(trend[:, 0], trend[:, 1], color="tomato")
            axes[1].axhline(0, color="gray")
            axes[1].set(xlabel="Linear fitted value", ylabel="Linear residual")
            fig.tight_layout()
            fig.savefig(target / "ch11-boston-curvature.svg")
            plt.close(fig)
            influence = linear.get_influence()
            internal = influence.resid_studentized_internal
            fig, axes = plt.subplots(2, 2, figsize=(9, 7))
            axes[0, 0].scatter(linear.fittedvalues, linear.resid, s=6)
            axes[0, 0].plot(trend[:, 0], trend[:, 1], color="tomato")
            axes[0, 0].set(xlabel="Fitted value", ylabel="Residual")
            theoretical = stats.norm.ppf(
                (np.arange(1, len(boston)+1)-0.5)/len(boston))
            axes[0, 1].scatter(theoretical, np.sort(internal), s=6)
            normal_quartiles = stats.norm.ppf([0.25, 0.75])
            response_quartiles = np.quantile(internal, [0.25, 0.75])
            qq_slope = np.diff(response_quartiles)[
                0]/np.diff(normal_quartiles)[0]
            qq_intercept = response_quartiles[0]-qq_slope*normal_quartiles[0]
            axes[0, 1].plot(theoretical, qq_intercept+qq_slope*theoretical)
            axes[0, 1].set(xlabel="Normal quantile",
                           ylabel="Standardized residual")
            axes[1, 0].scatter(linear.fittedvalues,
                               np.sqrt(np.abs(internal)), s=6)
            axes[1, 0].set(xlabel="Fitted value",
                           ylabel="sqrt(abs standardized residual)")
            axes[1, 1].scatter(influence.hat_matrix_diag, internal, s=6)
            axes[1, 1].set(xlabel="Leverage", ylabel="Standardized residual")
            fig.tight_layout()
            fig.savefig(target / "ch11-boston-diagnostics.svg")
            plt.close(fig)
    print("all assertions passed: equivalent formulas, centering, nested SSE")


if __name__ == "__main__":
    main()
