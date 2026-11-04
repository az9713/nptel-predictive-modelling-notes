"""Python weighting, diagnostics, and recovered source examples."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_goldfeldquandt

import matplotlib.pyplot as plt

matplotlib.use("Agg")


def home_workflow(frame):
    log_features = np.log(frame[["SqFeet", "Lot"]].to_numpy())
    log_response = np.log(frame["SalePrice"].to_numpy())
    design = sm.add_constant(log_features)
    ordinary = sm.OLS(log_response, design).fit()
    spread = sm.OLS(np.abs(ordinary.resid),
                    sm.add_constant(ordinary.fittedvalues)).fit().fittedvalues
    if np.any(spread <= 0):
        raise ValueError("Estimated standard deviations must remain positive")
    weighted = sm.WLS(log_response, design, weights=1 / spread**2).fit()
    return ordinary, weighted


def market_workflow(frame):
    design = sm.add_constant(frame[["Price", "P1", "P2"]])
    ordinary = sm.OLS(frame["MarketShare"], design).fit()
    discount = frame["Discount"].to_numpy()
    variances = [ordinary.resid[discount == group].var(ddof=1)
                 for group in [0, 1]]
    weights = np.array([1 / variances[int(group)] for group in discount])
    weighted = sm.WLS(frame["MarketShare"], design, weights=weights).fit()
    return ordinary, weighted, weights


def petroleum_workflow(frame):
    """Reusable conversion; frame requires the original source observations."""
    raw = sm.OLS(frame["consumption"], sm.add_constant(
        frame[["price", "vehicles"]])).fit()
    per_capita = sm.OLS(frame["consumption"] / frame["population"],
                        sm.add_constant(np.column_stack([
                            frame["price"],
                            frame["vehicles"] / frame["population"]
                        ]))).fit()
    return raw, per_capita


def expenditure_workflow(frame):
    """Raw/log-response fits without inventing missing original data."""
    design = sm.add_constant(frame["income"])
    if np.any(frame["expenditure"] <= 0):
        raise ValueError(
            "The log-response model requires positive expenditure")
    return (sm.OLS(frame["expenditure"], design).fit(),
            sm.OLS(np.log(frame["expenditure"]), design).fit())


def main():
    here = Path(__file__).resolve().parent
    estate = pd.read_csv(here / "data/ch15-homeprice-source.csv")
    market = pd.read_csv(here / "data/ch15-marketshare.txt", sep=r"\s+")
    home_raw = sm.OLS(estate["SalePrice"], sm.add_constant(
        estate[["SqFeet", "Lot"]])).fit()
    home_ols, home_wls = home_workflow(estate)
    market_ols, market_wls, weights = market_workflow(market)
    print("Recovered source-matching home and market examples; see provenance")
    print("Home rows:", len(estate))
    print("Home raw OLS:",
          np.array2string(home_raw.params.to_numpy(), precision=6))
    print("Home log OLS:", np.array2string(home_ols.params, precision=6))
    print("Home feasible WLS:", np.array2string(home_wls.params, precision=6))
    print("Market rows:", len(market))
    print(
        "Market OLS:",
        np.array2string(
            market_ols.params.to_numpy(),
            precision=6))
    print("Market group-variance WLS:",
          np.array2string(market_wls.params.to_numpy(), precision=6))
    rng = np.random.default_rng(1501)
    x = np.linspace(0.1, 3, 120)
    design = sm.add_constant(x)
    sd = 0.2 + x
    y = design @ np.array([2., 1.5]) + sd*rng.normal(size=len(x))
    ordinary = sm.OLS(y, design).fit()
    weighted = sm.WLS(y, design, weights=1 / sd**2).fit()
    direct = np.linalg.lstsq(design / sd[:, None], y / sd, rcond=None)[0]
    assert np.allclose(weighted.params, direct, atol=1e-12)
    bread = np.linalg.inv(design.T @ design)
    covariance_ols = bread @ (design.T @ (sd[:, None]**2*design)) @ bread
    covariance_gls = np.linalg.inv(design.T @ (design / sd[:, None]**2))
    eigenvalues = np.linalg.eigvalsh(covariance_ols - covariance_gls)
    assert eigenvalues.min() >= -1e-12
    statistic, pvalue, _ = het_goldfeldquandt(
        y, design, idx=1, split=50, drop=20, alternative="increasing",
        result_object=False)
    lower = sm.OLS(y[:50], design[:50]).fit()
    upper = sm.OLS(y[70:], design[70:]).fit()
    direct_f = upper.mse_resid / lower.mse_resid
    assert np.isclose(statistic, direct_f)
    assert np.isclose(pvalue, stats.f.sf(direct_f, 48, 48))
    print("Synthetic known-variance example; seed=1501, n=120")
    print(f"True slope variance: OLS={covariance_ols[1, 1]:.9f}, "
          f"GLS={covariance_gls[1, 1]:.9f}")
    print(f"GQ increasing F={statistic:.9f}, p={pvalue:.9g}, df=(48,48)")
    print("GLS/whitening and GQ/direct-F checks: passed")
    gasoline = pd.read_csv(here / "data/ch15-gas10.csv")
    petroleum = gasoline.rename(
        columns={
            "PCON": "consumption",
            "PRICE": "price",
            "REG": "vehicles",
            "POP": "population"})
    petroleum_raw, petroleum_capita = petroleum_workflow(petroleum)
    gas_design = sm.add_constant(petroleum[["price", "vehicles"]])
    gas_f, gas_p, _ = het_goldfeldquandt(
        petroleum["consumption"], gas_design, alternative="increasing",
        result_object=False)
    print("Publisher GAS10 rows:", len(gasoline))
    print("Petroleum total-model coefficients:",
          np.array2string(np.asarray(petroleum_raw.params), precision=9))
    print("Petroleum per-capita coefficients:",
          np.array2string(np.asarray(petroleum_capita.params), precision=9))
    print(f"Source-order GQ F={gas_f:.9f}, p={gas_p:.9f}")
    synthetic = pd.DataFrame({"income": x, "expenditure": np.exp(1 + x/3),
                              "population": 10 + 20*x,
                              "price": 1 + rng.uniform(size=len(x)),
                              "vehicles": (10 + 20*x)*(0.5 + np.sin(x)/8),
                              "consumption": (10 + 20*x)*(1 + x/5)})
    petroleum_workflow(synthetic)
    expenditure_workflow(synthetic)
    print("Petroleum/expenditure conversion functions: "
          "synthetic schema checks passed")
    print("Original expenditure CSV pending: "
          "no source-data expenditure results claimed")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(home_ols.fittedvalues, home_ols.resid, s=8, alpha=0.5)
    axes[0].set(xlabel="Fitted log price", ylabel="Raw log-scale residual")
    axes[1].scatter(market_wls.fittedvalues,
                    np.sqrt(weights)*market_wls.resid, c=market["Discount"])
    axes[1].set(xlabel="Fitted market share", ylabel="Whitened residual")
    for ax in axes:
        ax.axhline(0, color="gray", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(here / "ch15-diagnostics.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
