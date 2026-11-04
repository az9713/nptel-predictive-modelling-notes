"""Source-data stepwise selection and regression case studies in Python."""
from pathlib import Path
from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

import matplotlib.pyplot as plt

matplotlib.use("Agg")


def fit_columns(features, target, columns):
    design_columns = [np.ones(len(target))]
    design_columns.extend(features[:, j] for j in columns)
    design = np.column_stack(design_columns)
    return sm.OLS(target, design).fit()


def extract_aic(fit, penalty=2):
    """R extractAIC.lm, unknown scale, full-rank fixed-row comparison."""
    return fit.nobs*np.log(fit.ssr / fit.nobs) + penalty*fit.df_model + penalty


def stepwise(features, target, direction, penalty=2):
    """Single-column terms, fixed rows, no interactions or rank defects."""
    available = list(range(features.shape[1]))
    current = available.copy() if direction == "backward" else []
    fit = fit_columns(features, target, current)
    path = [(current.copy(), extract_aic(fit, penalty))]
    while True:
        options = []
        for column in (current if direction == "backward" else available):
            if direction == "forward" and column in current:
                continue
            proposed = ([j for j in current if j != column]
                        if direction == "backward" else current + [column])
            candidate = fit_columns(features, target, proposed)
            options.append((extract_aic(candidate, penalty),
                            proposed, candidate))
        if not options:
            break
        value, proposed, candidate = min(options, key=lambda item: item[0])
        if value >= path[-1][1] - 1e-10:
            break
        current, fit = proposed, candidate
        path.append((current.copy(), value))
    return fit, current, path


def case_study(here):
    women = pd.read_csv(here / "data/ch16-women.csv")
    print("women polynomial fits; rows:", len(women))
    for degree in [1, 2, 3]:
        design = np.vander(
            women["height"].to_numpy(),
            degree + 1,
            increasing=True)
        fitted = sm.OLS(women["weight"], design).fit()
        coefficients = np.array2string(np.asarray(fitted.params), precision=6)
        print(f"Degree {degree}: adjusted R2={fitted.rsquared_adj:.9f}, "
              f"coefficients={coefficients}")
    states = pd.read_csv(here / "data/ch16-state.x77.csv")
    names = ["Population", "Income", "Illiteracy", "Frost"]
    design = sm.add_constant(states[names])
    fit = sm.OLS(states["Murder"], design).fit()
    reduced = sm.OLS(states["Murder"], sm.add_constant(
        states[["Population", "Illiteracy"]])).fit()
    print("state.x77 full coefficients:",
          np.array2string(fit.params.to_numpy(), precision=9))
    print("State full 95% intervals:\n", fit.conf_int().round(6).to_string())
    print(
        "State VIF:", [
            round(
                variance_inflation_factor(
                    design.to_numpy(), j), 6) for j in range(
                1, len(names) + 1)])
    influence = fit.get_influence()
    student = influence.resid_studentized_external
    tail_probability = stats.t.sf(abs(student), fit.df_resid - 1)
    pvalues = np.minimum(1., 2*tail_probability*len(states))
    nevada = int(np.flatnonzero(states["rownames"] == "Nevada")[0])
    print(f"Nevada external studentized residual={student[nevada]:.9f}, "
          f"Bonferroni p={pvalues[nevada]:.9f}")
    print(f"State alphabetical-order DW={durbin_watson(fit.resid):.9f}")
    print("State ordering is not a time index; "
          "serial interpretation is unsupported.")
    rng = np.random.default_rng(1602)
    simulated_y = np.asarray(fit.fittedvalues)[:, None]
    simulated_y = simulated_y + rng.choice(np.asarray(fit.resid),
                                           size=(len(states), 1000))
    fitted_all = design.to_numpy() @ np.linalg.lstsq(
        design.to_numpy(), simulated_y, rcond=None)[0]
    simulated_residual = simulated_y - fitted_all
    simulated_dw = np.sum(np.diff(simulated_residual, axis=0)**2, axis=0)
    simulated_dw /= np.sum(simulated_residual**2, axis=0)
    tail = np.mean(simulated_dw <= durbin_watson(fit.resid))
    simulated_p = min(1., 2*min(tail, 1-tail))
    print("Residual-resampling two-sided DW p (1000 draws)="
          f"{simulated_p:.6f}")
    ftest, pvalue, difference = fit.compare_f_test(reduced)
    print(f"Full versus reduced F={ftest:.9f}, p={pvalue:.9f}, "
          f"df difference={difference:.0f}")
    _, selected, _ = stepwise(
        states[names].to_numpy(), states["Murder"].to_numpy(), "backward")
    print("State backward retained:", [names[j] for j in selected])
    target = states["Murder"].to_numpy()

    def neg_profile(power):
        transformed = (np.log(target) if abs(power) < 1e-8
                       else np.expm1(power*np.log(target)) / power)
        rss = sm.OLS(transformed, design).fit().ssr
        return len(target)*np.log(rss / len(target)) / \
            2 - (power - 1)*np.log(target).sum()

    power = optimize.minimize_scalar(neg_profile, bounds=(-2, 2),
                                     method="bounded").x
    print(f"Conditional Gaussian-regression Box-Cox profile power={power:.6f}")
    marginal_power = stats.boxcox_normmax(target, method="mle")
    print(f"Marginal normal-family Box-Cox power={marginal_power:.6f}")
    spread_model = sm.RLM(
        np.log(np.abs(influence.resid_studentized_internal)),
        sm.add_constant(np.log(np.asarray(fit.fittedvalues)))).fit()
    suggested_power = 1 - spread_model.params[1]
    print("Robust spread-level suggested response "
          f"power={suggested_power:.6f}")
    best_subsets = []
    for count in range(1, len(names) + 1):
        alternatives = []
        for subset in combinations(range(len(names)), count):
            candidate = fit_columns(states[names].to_numpy(), target, subset)
            cp = candidate.ssr / fit.mse_resid - len(states) + 2*(count + 1)
            alternatives.append((cp, subset, candidate.rsquared_adj))
        best = min(alternatives, key=lambda item: item[0])
        best_subsets.append(best)
        print("Best size", count, "Cp:", round(best[0], 6),
              "terms:", [names[j] for j in best[1]])
    partial_figure, partial_axes = plt.subplots(2, 2, figsize=(10, 7))
    for column, ax in zip(names, partial_axes.flat):
        x_values = states[column].to_numpy()
        component = fit.resid + fit.params[column]*x_values
        ax.scatter(x_values, component)
        ax.plot(x_values[np.argsort(x_values)],
                fit.params[column]*np.sort(x_values), color="firebrick")
        smooth = sm.nonparametric.lowess(component, x_values, frac=2/3)
        ax.plot(smooth[:, 0], smooth[:, 1], color="black")
        ax.set(xlabel=column, ylabel="Component plus residual")
    partial_figure.tight_layout()
    partial_figure.savefig(here / "ch16-component-plots.svg")
    plt.close(partial_figure)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].scatter(fit.fittedvalues, fit.resid)
    axes[0, 0].set(xlabel="Fitted murder rate", ylabel="Residual")
    probabilities = (np.arange(1, len(states) + 1) - 0.5) / len(states)
    theoretical = stats.t.ppf(probabilities, fit.df_resid - 1)
    ordered = np.sort(student)
    robust = sm.RLM(ordered, sm.add_constant(theoretical)).fit()
    simulated_y = np.asarray(fit.fittedvalues)[:, None]
    simulated_y = simulated_y + np.sqrt(fit.mse_resid)*rng.normal(
        size=(len(states), 100))
    residual_sim = simulated_y - design.to_numpy() @ np.linalg.lstsq(
        design.to_numpy(), simulated_y, rcond=None)[0]
    leverage = influence.hat_matrix_diag[:, None]
    residual_sse = np.sum(residual_sim**2, axis=0)
    adjustment = residual_sim**2 / (1 - leverage)
    deleted_scale = (residual_sse - adjustment) / (fit.df_resid - 1)
    simulated_t = residual_sim / np.sqrt(deleted_scale*(1 - leverage))
    envelope = np.quantile(np.sort(simulated_t, axis=0), [0.025, 0.975],
                           axis=1, method="linear")
    axes[0, 1].fill_between(theoretical, envelope[0], envelope[1], alpha=0.2)
    axes[0, 1].scatter(theoretical, ordered, s=14)
    axes[0, 1].plot(theoretical, robust.predict(sm.add_constant(theoretical)),
                    color="firebrick")
    axes[0, 1].set_title("Studentized residuals; simulated pointwise envelope")
    axes[1, 0].scatter(fit.fittedvalues,
                       np.sqrt(abs(influence.resid_studentized_internal)))
    axes[1, 0].set(xlabel="Fitted murder rate",
                   ylabel="sqrt(abs standardized residual)")
    axes[1, 1].scatter(influence.hat_matrix_diag, student,
                       s=20 + 300*influence.cooks_distance[0])
    axes[1, 1].set(xlabel="Leverage", ylabel="Externally studentized residual")
    for state in ["Alaska", "Hawaii", "Nevada"]:
        index = int(np.flatnonzero(states["rownames"] == state)[0])
        axes[1, 1].annotate(
            state, (influence.hat_matrix_diag[index], student[index]))
    figure.tight_layout()
    figure.savefig(here / "ch16-state-diagnostics.svg")
    plt.close(figure)


def main():
    here = Path(__file__).resolve().parent
    pollution = pd.read_csv(here / "data/ch16-pollution.csv")
    features = pollution.iloc[:, :15].to_numpy()
    target = pollution["mort"].to_numpy()
    full_fit = fit_columns(features, target, list(range(15)))
    print("pollution full extractAIC:", f"{extract_aic(full_fit):.9f}")
    for direction in ["backward", "forward"]:
        fit, selected, path = stepwise(features, target, direction)
        print(direction, "step path:")
        for columns, value in path:
            print(" ", [j + 1 for j in columns], f"{value:.9f}")
        print(f"Final adjusted R2={fit.rsquared_adj:.9f}")
        expected = ([1, 2, 3, 4, 5, 6, 9, 12, 13] if direction == "backward"
                    else [9, 6, 2, 14, 1, 3, 5])
        assert [j + 1 for j in selected] == expected
    train, test = train_test_split(np.arange(len(target)), test_size=0.25,
                                   random_state=1601)
    train_fit, selected, _ = stepwise(
        features[train], target[train], "backward")
    test_columns = [np.ones(len(test))]
    test_columns.extend(features[test, j] for j in selected)
    test_design = np.column_stack(test_columns)
    error = mean_squared_error(target[test], train_fit.predict(test_design))
    print(f"Authored held-out example MSE={error:.6f}")
    case_study(here)
    print("Source-step-path assertions and diagnostics: passed")


if __name__ == "__main__":
    main()
