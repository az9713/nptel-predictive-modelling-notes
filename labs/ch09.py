"""Python conversion of the robustbase delivery outlier demonstration.

Data: 25 factual triples from robustbase::delivery,
distributed by the Rdatasets mirror; primary CRAN manual identifies the same
variables, dimensions, and Montgomery-Peck source. No R runtime is required;
fresh-R comparisons are in validation/chapters/ch07-09-r-comparison.json.
"""
import argparse
from pathlib import Path

import numpy as np
import statsmodels.api as sm

DATA = np.array([
    [7, 560, 16.68], [3, 220, 11.50], [3, 340, 12.03], [4, 80, 14.88],
    [6, 150, 13.75], [7, 330, 18.11], [2, 110, 8.00], [7, 210, 17.83],
    [30, 1460, 79.24], [5, 605, 21.50], [16, 688, 40.33], [10, 215, 21.00],
    [4, 255, 13.50], [6, 462, 19.75], [9, 448, 24.00], [10, 776, 29.00],
    [6, 200, 15.35], [7, 132, 19.00], [3, 36, 9.50], [17, 770, 35.10],
    [10, 140, 17.90], [26, 810, 52.32], [9, 450, 18.75], [8, 635, 19.83],
    [4, 150, 10.75],
])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots", help="Output folder for Python diagnostic plots")
    args = parser.parse_args()
    np.set_printoptions(precision=9, suppress=True)
    design = sm.add_constant(DATA[:, :2])
    y = DATA[:, 2]
    fit = sm.OLS(y, design).fit()
    influence = fit.get_influence()
    h = influence.hat_matrix_diag
    e = fit.resid
    n, q = design.shape
    inverse = np.linalg.inv(design.T @ design)
    internal = e/np.sqrt(fit.mse_resid*(1-h))
    deleted_variance = ((n-q)*fit.mse_resid-e**2/(1-h))/(n-q-1)
    external = e/np.sqrt(deleted_variance*(1-h))
    cook = internal**2*h/(q*(1-h))
    dffits = external*np.sqrt(h/(1-h))
    np.testing.assert_allclose(internal, influence.resid_studentized_internal)
    np.testing.assert_allclose(external, influence.resid_studentized_external)
    np.testing.assert_allclose(cook, influence.cooks_distance[0])
    np.testing.assert_allclose(dffits, influence.dffits[0])
    np.testing.assert_allclose(h.sum(), q)
    for i in range(n):
        keep = np.arange(n) != i
        deleted = sm.OLS(y[keep], design[keep]).fit()
        change = inverse @ design[i]*e[i]/(1-h[i])
        np.testing.assert_allclose(
            fit.params-deleted.params, change, atol=1e-10)
        np.testing.assert_allclose(y[i]-design[i] @ deleted.params,
                                   e[i]/(1-h[i]), atol=1e-10)
        dfbetas = change/np.sqrt(deleted.mse_resid*np.diag(inverse))
        np.testing.assert_allclose(dfbetas, influence.dfbetas[i], atol=1e-10)
    print("data: robustbase delivery, original order, 25 rows")
    print("intercept products distance:", np.round(fit.params, 9))
    print("n q df SSE:", n, q, int(fit.df_resid), round(fit.ssr, 9))
    print("leverage_sum mean:", np.round([h.sum(), h.mean()], 9))
    print("Cook screening threshold4/n:", round(4/n, 9))
    print("Cook-screen rows,1-based:", (np.flatnonzero(cook > 4/n)+1).tolist())
    for i in [8, 21]:
        diagnostic = [h[i], e[i], internal[i], external[i], cook[i], dffits[i]]
        print("row leverage residual internal external Cook DFFITS:", i+1,
              np.round(diagnostic, 9))
        print("row DFBETAS intercept products distance:", i+1,
              np.round(influence.dfbetas[i], 9))
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        target = Path(args.plots)
        target.mkdir(parents=True, exist_ok=True)
        rows = np.arange(1, n+1)
        panels = [(external, "Externally studentized residual", [-2, 2]),
                  (internal, "Internally standardized residual", [-2, 2]),
                  (h, "Leverage", [2*q/n]),
                  (dffits, "DFFITS", [-2*np.sqrt(q/n), 2*np.sqrt(q/n)]),
                  (cook, "Cook distance", [4/n])]
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(3, 2, figsize=(10, 10))
            for axis, (values, label, cutoffs) in zip(axes.flat, panels):
                axis.stem(rows, values)
                axis.set_title(label)
                axis.set_xlabel("Observation, original1-based order")
                for cutoff in cutoffs:
                    axis.axhline(cutoff, color="tomato", linestyle="--")
                for row in [9, 22]:
                    axis.annotate(str(row), (row, values[row-1]))
            axes.flat[-1].plot(rows, cook, "o-")
            axes.flat[-1].axhline(4/n, color="tomato", linestyle="--")
            axes.flat[-1].set_title("Cook distance, chart form")
            axes.flat[-1].set_xlabel("Observation")
            fig.tight_layout()
            fig.savefig(target / "ch09-delivery-diagnostics.svg")
            plt.close(fig)
            fig, axes = plt.subplots(3, 1, figsize=(8, 8))
            for j, axis in enumerate(axes):
                axis.stem(rows, influence.dfbetas[:, j])
                axis.set_title(["Intercept", "Products", "Distance"][j])
                for cutoff in [-2/np.sqrt(n), 2/np.sqrt(n)]:
                    axis.axhline(cutoff, color="tomato", linestyle="--")
            axes[-1].set_xlabel("Observation, original1-based order")
            fig.tight_layout()
            fig.savefig(target / "ch09-delivery-dfbetas.svg")
            plt.close(fig)
    print("all assertions passed, including all25 explicit deleted refits")


if __name__ == "__main__":
    main()
