"""Convert ISLR Default and Smarket binomial-logit demonstrations."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

matplotlib.use("Agg")


def likelihood(beta, design, target):
    linear = design @ beta
    return target @ linear - np.logaddexp(0, linear).sum()


def newton_logit(design, target):
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        probability = expit(design @ beta)
        gradient = design.T @ (target - probability)
        weight = probability*(1 - probability)
        information = design.T @ (weight[:, None]*design)
        step = np.linalg.solve(information, gradient)
        scale = 1.
        previous = likelihood(beta, design, target)
        while likelihood(beta + scale*step, design, target) < previous:
            scale /= 2
            if scale < 1e-10:
                raise RuntimeError("Logistic Newton line search failed")
        beta += scale*step
        if np.max(abs(scale*step)) < 1e-9:
            return beta
    raise RuntimeError("Logistic Newton fit did not converge")


def main():
    here = Path(__file__).resolve().parent
    default = pd.read_csv(here / "data/ch17-default.csv")
    y = (default["default"] == "Yes").to_numpy(dtype=float)
    student = (default["student"] == "Yes").to_numpy(dtype=float)
    balance = default["balance"].to_numpy()
    income = default["income"].to_numpy()
    balance_design = sm.add_constant(balance)
    balance_fit = sm.GLM(
        y,
        balance_design,
        family=sm.families.Binomial()).fit(
        tol=1e-12)
    student_design = sm.add_constant(student)
    student_fit = sm.GLM(
        y,
        student_design,
        family=sm.families.Binomial()).fit(
        tol=1e-12)
    full_design = np.column_stack(
        [np.ones(len(y)), student, balance/1000, income/10000])
    full = sm.GLM(y, full_design, family=sm.families.Binomial()).fit(tol=1e-12)
    independent = newton_logit(full_design, y)
    assert np.allclose(independent, full.params, atol=1e-7)
    raw_coefficient = full.params / np.array([1, 1, 1000, 10000])
    print("ISLR Default is simulated; rows:",
          len(default), "defaults:", int(y.sum()))
    print(
        "Balance-only coefficients:",
        np.array2string(
            balance_fit.params,
            precision=9))
    prediction = balance_fit.predict([[1, 1000], [1, 2000]])
    print("Balance1000/2000 default probabilities:",
          np.array2string(prediction, precision=9))
    print(
        "Student-only coefficients:",
        np.array2string(
            student_fit.params,
            precision=9))
    print("Student0/1 marginal probabilities:",
          np.array2string(student_fit.predict([[1, 0], [1, 1]]), precision=9))
    print("Adjusted coefficients (const,student,balance,income):",
          np.array2string(raw_coefficient, precision=9))
    print(f"Default null deviance={full.null_deviance:.9f}, "
          f"residual={full.deviance:.9f}, AIC={full.aic:.9f}")
    print("Independent Newton/GLM coefficient agreement: passed")
    assert np.isclose(full.fittedvalues.mean(), y.mean(), atol=1e-10)
    probe = np.array([-0.5, 0.2, 0.8, -0.1])
    probability = expit(full_design @ probe)
    analytic = full_design.T @ (y - probability)
    h = 1e-5
    numeric = []
    for direction in np.eye(4):
        upper = likelihood(probe + h*direction, full_design, y)
        lower = likelihood(probe - h*direction, full_design, y)
        numeric.append((upper - lower) / (2*h))
    numeric = np.array(numeric)
    assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-5)
    print("Finite-difference likelihood score check: passed")
    cells = np.array([[688., 650.], [21., 59.]])
    odds_ratio = cells[0, 0]*cells[1, 1] / (cells[0, 1]*cells[1, 0])
    standard_error = np.sqrt(np.sum(1/cells))
    print(f"Source illustrative table OR={odds_ratio:.9f}, "
          f"logOR SE={standard_error:.9f}")
    market = pd.read_csv(here / "data/ch17-smarket.csv")
    columns = [f"Lag{i}" for i in range(1, 6)] + ["Volume"]
    market_y = (market["Direction"] == "Up").to_numpy(dtype=int)
    market_design = sm.add_constant(market[columns])
    fit = sm.GLM(
        market_y,
        market_design,
        family=sm.families.Binomial()).fit(
        tol=1e-12)
    matrix = confusion_matrix(market_y, fit.predict() > 0.5, labels=[0, 1])
    improvement = fit.null_deviance - fit.deviance
    print(
        "Smarket coefficients:",
        np.array2string(
            fit.params.to_numpy(),
            precision=9))
    lr_pvalue = stats.chi2.sf(improvement, 6)
    print(f"Smarket null deviance={fit.null_deviance:.9f}, "
          f"residual={fit.deviance:.9f}, LR={improvement:.9f}, "
          f"p={lr_pvalue:.9f}")
    print("Confusion matrix rows=actualDown/Up, "
          "columns=predictedDown/Up:\n", matrix)
    accuracy = np.trace(matrix) / matrix.sum()
    error = 1 - accuracy
    print(f"Training accuracy={accuracy:.6f}, error={error:.6f}, "
          f"always-Up={market_y.mean():.6f}")
    assert np.array_equal(matrix, [[145, 457], [141, 507]])
    train = market["Year"] < 2005
    lag_design = sm.add_constant(market[["Lag1", "Lag2"]])
    held_fit = sm.GLM(market_y[train], lag_design[train],
                      family=sm.families.Binomial()).fit(tol=1e-12)
    held_prediction = held_fit.predict(lag_design[~train]) > 0.5
    held_matrix = confusion_matrix(
        market_y[~train], held_prediction, labels=[0, 1])
    print("Authored chronological2005 lag-only confusion:\n", held_matrix)
    held_accuracy = np.trace(held_matrix) / held_matrix.sum()
    print(f"Chronological2005 accuracy={held_accuracy:.9f}")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([balance[y == 0], balance[y == 1]],
                    tick_labels=["No", "Yes"])
    axes[0].set(xlabel="Default", ylabel="Balance")
    grid = np.linspace(0, 2500, 150)
    for flag in [0, 1]:
        design = np.column_stack([np.ones(len(grid)), np.full(len(grid), flag),
                                  grid/1000, np.full(len(grid), 3.)])
        axes[1].plot(grid, full.predict(design), label=f"student={flag}")
    axes[1].set(
        xlabel="Balance, income fixed at30000",
        ylabel="Fitted probability")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(here / "ch17-default.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
