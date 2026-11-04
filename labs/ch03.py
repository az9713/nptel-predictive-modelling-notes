"""Principal-axis factor analysis and conversion of all source R examples."""

from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

from ch02 import varimax

DATA = Path(__file__).resolve().parent / "data"


def covariance_to_correlation(covariance):
    """Match stats::cov2cor for a positive-diagonal covariance matrix."""
    scales = np.sqrt(np.diag(covariance))
    return np.asarray(covariance) / np.outer(scales, scales)


def principal_axis(correlation, count=2, tolerance=0.001, max_iter=50):
    """psych fm='pa': SMC start; total-communality stopping rule."""
    matrix = np.asarray(correlation, dtype=float)
    communalities = 1 - 1 / np.diag(np.linalg.inv(matrix))
    reduced = matrix.copy()
    np.fill_diagonal(reduced, communalities)
    previous = communalities.sum()
    history = []
    for _ in range(max_iter):
        values, vectors = np.linalg.eigh(reduced)
        values, vectors = values[::-1], vectors[:, ::-1]
        if np.any(values[:count] < 0):
            raise ValueError(
                "A retained principal-axis eigenvalue is negative"
            )
        loadings = vectors[:, :count] * np.sqrt(values[:count])
        communalities = np.sum(loadings**2, axis=1)
        total = communalities.sum()
        history.append(total)
        difference = abs(total - previous)
        np.fill_diagonal(reduced, communalities)
        previous = total
        if difference <= tolerance:
            break
    for column in range(count):
        if loadings[:, column].sum() < 0:
            loadings[:, column] *= -1
    order = np.argsort(np.sum(loadings**2, axis=0))[::-1]
    return loadings[:, order], np.array(history)


def common_eigenvalues(correlation, use_smc=True):
    """One-factor MINRES reduced spectrum for parallel FA."""
    model = FactorAnalyzer(
        n_factors=1,
        rotation=None,
        method="minres",
        is_corr_matrix=True,
        use_smc=use_smc,
    )
    model.fit(correlation)
    reduced = np.asarray(correlation).copy()
    np.fill_diagonal(reduced, model.get_communalities())
    return np.linalg.eigvalsh(reduced)[::-1]


def parallel_both(correlation, observations, iterations=100, seed=2026):
    """Convert matrix-input fa.parallel(...,fa='both') with explicit n.obs."""
    matrix = np.asarray(correlation)
    rng = np.random.default_rng(seed)
    pc_null, fa_null = [], []
    for _ in range(iterations):
        sample = rng.normal(size=(observations, len(matrix)))
        null = np.corrcoef(sample, rowvar=False)
        pc_null.append(np.linalg.eigvalsh(null)[::-1])
        fa_null.append(common_eigenvalues(null, use_smc=False))
    return {
        "pc_actual": np.linalg.eigvalsh(matrix)[::-1],
        "fa_actual": common_eigenvalues(matrix),
        "pc_cutoff": np.quantile(pc_null, 0.95, axis=0),
        "fa_cutoff": np.quantile(fa_null, 0.95, axis=0),
    }


def plot_factor_examples(
    stock, unrotated, rotated, ability_parallel, stock_parallel
):
    """Show preserved model covariance and the source factor comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    figure, axes = plt.subplots(
        1, 3, figsize=(13, 3.8), constrained_layout=True
    )
    for row, name in enumerate(stock.index):
        axes[0].scatter(*unrotated[row], marker="x", color="#fbbf24")
        axes[0].scatter(*rotated[row], marker="o", color="#6ee7d8")
        axes[0].annotate(
            name,
            rotated[row],
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0].axhline(0, color="#94a3b8", linewidth=0.5)
    axes[0].axvline(0, color="#94a3b8", linewidth=0.5)
    axes[0].set(
        xlabel="Factor1 loading",
        ylabel="Factor2 loading",
        title="Stock loading rotation: crosses to dots",
    )
    axes[0].set_xlim(-0.1, 1.1)
    axes[0].set_ylim(-0.55, 1.15)
    for axis, result, title in zip(
        axes[1:],
        [ability_parallel, stock_parallel],
        ["Ability: n.obs=112", "Stock: assumed n.obs=100"],
    ):
        positions = np.arange(1, len(result["pc_actual"]) + 1)
        for key, style, color in [
            ("pc_actual", "o-", "#6ee7d8"),
            ("fa_actual", "s-", "#c4b5fd"),
            ("pc_cutoff", "--", "#fbbf24"),
            ("fa_cutoff", ":", "#fb7185"),
        ]:
            axis.plot(positions, result[key], style, color=color, label=key)
        axis.set(xlabel="Ordered eigenvalue", ylabel="Eigenvalue", title=title)
        axis.legend(fontsize=7)
    figure.savefig(
        DATA.parents[1] / "assets" / "ch03-factor-examples.svg",
        metadata={"Date": None},
    )
    figure.savefig(
        DATA.parents[1] / "validation" / "chapters" / "ch03-figure.png",
        dpi=120,
    )
    plt.close(figure)


def main():
    np.set_printoptions(precision=6, suppress=True)
    fixture = np.array([[0.8, 0], [0.6, 0], [0, 0.7]])
    fixture_cov = fixture @ fixture.T + np.diag([0.36, 0.64, 0.51])
    assert np.allclose(fixture_cov, [[1, 0.48, 0], [0.48, 1, 0], [0, 0, 1]])
    print("Three-variable exercise covariance:")
    print(fixture_cov)
    ability_cov = pd.read_csv(
        DATA / "ch03-ability-covariance.csv", index_col=0
    )
    ability = covariance_to_correlation(ability_cov)
    stock = pd.read_csv(DATA / "ch03-stock-correlation.csv", index_col=0)
    print("Ability correlation matrix (converted cov2cor):")
    print(ability)
    results = {}
    for name, correlation in [
        ("ability", ability),
        ("stock", stock.to_numpy()),
    ]:
        loadings, history = principal_axis(correlation)
        rotated, rotation = varimax(loadings)
        for column in range(2):
            if rotated[:, column].sum() < 0:
                rotated[:, column] *= -1
        order = np.argsort(np.sum(rotated**2, axis=0))[::-1]
        rotated = rotated[:, order]
        common = loadings @ loadings.T
        uniqueness = np.diag(correlation - common)
        assert np.allclose(common, rotated @ rotated.T, atol=1e-12)
        assert np.allclose(rotation.T @ rotation, np.eye(2))
        assert np.all(uniqueness > 0)
        offdiag = ~np.eye(len(correlation), dtype=bool)
        rms = np.sqrt(np.mean((correlation - common)[offdiag] ** 2))
        results[name] = (loadings, rotated)
        print(f"{name} principal-axis iterations: {len(history)}")
        change = abs(history[-1] - history[-2])
        print(f"{name} final total-communality change: {change:.8f}")
        print(f"{name} source tolerance reached: {change <= 0.001}")
        print(f"{name} unrotated loadings:")
        print(loadings)
        print(f"{name} Kaiser-normalized varimax loadings:")
        print(rotated)
        print(f"{name} communalities:", np.sum(loadings**2, axis=1))
        print(f"{name} uniquenesses:", uniqueness)
        print(f"{name} off-diagonal RMS residual: {rms:.8f}")
        print(
            f"{name} score weights R^-1 L (individual rows unavailable):"
        )
        print(np.linalg.solve(correlation, rotated))
    ability_parallel = parallel_both(ability, 112)
    stock_parallel = parallel_both(stock, 100)
    for name, result in [
        ("ability", ability_parallel),
        ("stock", stock_parallel),
    ]:
        for key, values in result.items():
            print(name, key, values)
    loadings, rotated = results["stock"]
    plot_factor_examples(
        stock, loadings, rotated, ability_parallel, stock_parallel
    )
    continued, _ = principal_axis(stock, tolerance=1e-8, max_iter=100)
    continued_unique = 1 - np.sum(continued**2, axis=1)
    print("Stock uniqueness after100 diagnostic iterations:", continued_unique)
    assert continued_unique[3] < 0
    print(
        "Stock n unknown:100 is the source package fallback, not known n."
    )
    print(
        "Individual factor scores unavailable from summary matrices."
    )
    print(
        "Checks passed: covariance, principal-axis extraction, rotation."
    )


if __name__ == "__main__":
    main()
