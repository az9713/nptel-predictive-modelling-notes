"""Convert source ISLR Credit categorical-predictor examples to Python."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    here = Path(__file__).resolve().parent
    data = pd.read_csv(here / "data/ch10-credit.csv")
    data["Gender"] = pd.Categorical(data["Gender"].str.strip(),
                                    categories=["Male", "Female"])
    data["Student"] = pd.Categorical(data["Student"], categories=["No", "Yes"])
    data["Ethnicity"] = pd.Categorical(
        data["Ethnicity"],
        categories=["African American", "Asian", "Caucasian"])
    assert len(data) == 400
    print("data: ISLR Credit, 400 rows; Gender whitespace normalized")
    for name in ["Student", "Gender", "Ethnicity"]:
        fit = smf.ols(f"Balance ~ C({name})", data=data, missing="raise").fit()
        means = data.groupby(name, observed=True).Balance.mean()
        labels = data[name].cat.categories
        expected = np.r_[means.iloc[0],
                         means.iloc[1:].to_numpy()-means.iloc[0]]
        np.testing.assert_allclose(fit.params.to_numpy(), expected, atol=1e-10)
        print(name, "levels:", list(labels))
        print(name, "group_means:", np.round(means.to_numpy(), 9))
        print(name, "coefficients:", np.round(fit.params.to_numpy(), 9))
    female = (data.Gender == "Female").astype(float).to_numpy()
    y = data.Balance.to_numpy()
    fitted = []
    for name, code in [("female1_male0", female),
                       ("male1_female0", 1-female),
                       ("female_plus1_male_minus1", 2*female-1)]:
        fit = sm.OLS(y, sm.add_constant(code)).fit()
        print(name, "coefficients:", np.round(fit.params, 9))
        fitted.append(fit.fittedvalues)
    np.testing.assert_allclose(fitted[0], fitted[1], atol=1e-10)
    np.testing.assert_allclose(fitted[0], fitted[2], atol=1e-10)
    indicators = pd.get_dummies(data.Ethnicity, dtype=float).to_numpy()
    redundant = np.column_stack([np.ones(len(data)), indicators])
    print("three_categories: redundant_columns rank:", redundant.shape[1],
          np.linalg.matrix_rank(redundant))
    reduced = np.column_stack([np.ones(len(data)), indicators[:, 1:]])
    fit1 = np.linalg.lstsq(reduced, y, rcond=None)[0]
    fit2 = np.linalg.lstsq(indicators, y, rcond=None)[0]
    np.testing.assert_allclose(reduced @ fit1, indicators @ fit2, atol=1e-10)
    print("all assertions passed: group means, coding invariance, rank")


if __name__ == "__main__":
    main()
