"""PCA and explicit Python conversions of the source R demonstrations."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def varimax(loadings, normalize=True, tolerance=1e-5, max_iter=1000):
    """Varimax with Kaiser normalization and polar updates."""
    original = np.asarray(loadings, dtype=float)
    norms = (
        np.linalg.norm(original, axis=1)
        if normalize
        else np.ones(len(original))
    )
    norms = np.where(norms == 0, 1, norms)
    normalized = original / norms[:, None]
    rotation = np.eye(normalized.shape[1])
    previous = 0.0
    for _ in range(max_iter):
        trial = normalized @ rotation
        target = trial**3 - trial @ np.diag(np.sum(trial**2, axis=0)) / len(
            trial
        )
        left, singular, right = np.linalg.svd(normalized.T @ target)
        rotation = left @ right
        objective = singular.sum()
        if previous and objective <= previous * (1 + tolerance):
            break
        previous = objective
    return original @ rotation, rotation


def principal(correlation, count, rotation="none"):
    """Correlation-PCA loadings and psych-style standardized-score weights."""
    matrix = np.asarray(correlation, dtype=float)
    values, vectors = np.linalg.eigh(matrix)
    values, vectors = values[::-1], vectors[:, ::-1]
    loadings = vectors[:, :count] * np.sqrt(np.maximum(values[:count], 0))
    if rotation == "varimax":
        loadings, _ = varimax(loadings)
    for column in range(count):
        if loadings[:, column].sum() < 0:
            loadings[:, column] *= -1
    order = np.argsort(np.sum(loadings**2, axis=0))[::-1]
    loadings = loadings[:, order]
    weights = np.linalg.solve(matrix, loadings)
    return values, loadings, weights


def parallel_pca(
    correlation, observations, raw=None, iterations=100, seed=2026
):
    """PCA parallel analysis: Gaussian null and optional column bootstrapping.

    This converts the documented psych::fa.parallel(fa='pc') mechanism.
    RNG streams differ from R; Monte Carlo curves are not bitwise copies.
    """
    rng = np.random.default_rng(seed)
    dimension = len(correlation)
    simulated = np.empty((iterations, dimension))
    resampled = np.empty_like(simulated) if raw is not None else None
    for iteration in range(iterations):
        null = rng.normal(size=(observations, dimension))
        simulated[iteration] = np.linalg.eigvalsh(
            np.corrcoef(null, rowvar=False)
        )[::-1]
        if raw is not None:
            draws = np.column_stack(
                [
                    rng.choice(raw[:, j], observations, replace=True)
                    for j in range(dimension)
                ]
            )
            resampled[iteration] = np.linalg.eigvalsh(
                np.corrcoef(draws, rowvar=False)
            )[::-1]
    actual = np.linalg.eigvalsh(correlation)[::-1]
    cutoff = np.quantile(simulated, 0.95, axis=0)
    return actual, cutoff, resampled


def plot_source_examples(judges, values, cutoff, bootstrap, harman, h_cutoff):
    """Render the source scatter and scree diagrams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    figure, axes = plt.subplots(
        1, 3, figsize=(13, 3.7), constrained_layout=True
    )
    axes[0].scatter(judges["DECI"], judges["DILG"], s=18, color="#6ee7d8")
    axes[0].set(
        xlabel="DECI: prompt decisions",
        ylabel="DILG: diligence",
        title="Source paired observations",
    )
    for ax, observed, threshold in [
        (axes[1], values, cutoff),
        (axes[2], harman, h_cutoff),
    ]:
        positions = np.arange(1, len(observed) + 1)
        ax.plot(positions, observed, "o-", color="#6ee7d8", label="Observed")
        ax.plot(
            positions, threshold, "--", color="#fbbf24", label="Gaussian 95%"
        )
        ax.axhline(1, color="#94a3b8", linewidth=0.7)
        ax.set(xlabel="Ordered component", ylabel="Eigenvalue")
        ax.legend(fontsize=7)
    axes[1].plot(
        np.arange(1, len(values) + 1),
        np.quantile(bootstrap, 0.95, axis=0),
        ":",
        color="#fb7185",
        label="Column bootstrap 95%",
    )
    axes[1].legend(fontsize=7)
    axes[1].set_title("Judge ratings: 43 rows, 11 features")
    axes[2].set_title("Harman: source n.obs=302")
    destination = DATA.parents[1] / "assets" / "ch02-source-plots.svg"
    figure.savefig(destination, metadata={"Date": None})
    figure.savefig(
        DATA.parents[1] / "validation" / "chapters" / "ch02-figure.png",
        dpi=120,
    )
    plt.close(figure)


def main():
    np.set_printoptions(precision=6, suppress=True)
    judges = pd.read_csv(DATA / "ch02-USJudgeRatings.csv", index_col=0)
    ratings = judges.drop(columns="CONT")
    raw = ratings.to_numpy()
    centered = raw - raw.mean(axis=0)
    standardized = centered / raw.std(axis=0, ddof=1)
    correlation = np.corrcoef(raw, rowvar=False)
    values, loadings, weights = principal(correlation, 1)
    scores = standardized @ weights
    raw_scores = standardized @ (loadings / np.sqrt(values[0]))
    assert np.allclose(np.cov(scores, rowvar=False), 1)
    assert np.allclose(raw_scores / np.sqrt(values[0]), scores)
    assert np.allclose(correlation @ weights, loadings)
    assert np.allclose(loadings[:, 0] ** 2, np.sum(loadings**2, axis=1))
    print("USJudgeRatings canonical shape:", judges.shape)
    print("Source rating-only PCA omits numeric CONT, leaving:", ratings.shape)
    print("DECI/DILG covariance (n-1), matching source column order:")
    print(judges[["DECI", "DILG"]].cov().to_numpy())
    print("DILG/DECI correlation:", judges[["DILG", "DECI"]].corr().iloc[0, 1])
    print("Rating correlation eigenvalues:", values)
    print(
        f"First-component fraction of rating variance: {values[0] / 11:.8f}"
    )
    print("Feature, loading, communality, uniqueness, score weight:")
    for name, loading, weight in zip(
        ratings.columns, loadings[:, 0], weights[:, 0]
    ):
        print(
            f"{name:4s} {loading:.6f} {loading**2:.6f} "
            f"{1 - loading**2:.6f} {weight:.6f}"
        )
    print("First six psych-style standardized scores:", scores[:6, 0])
    actual, cutoff, boot = parallel_pca(correlation, len(raw), raw)
    print("Parallel PCA Gaussian 95% null eigenvalues (100 draws):", cutoff)
    print(
        "Parallel PCA bootstrap 95% null eigenvalues:",
        np.quantile(boot, 0.95, axis=0),
    )
    print("Count above Gaussian null:", int(np.sum(actual > cutoff)))
    two = judges[["DECI", "DILG"]].to_numpy()
    two -= two.mean(axis=0)
    _, singular, right = np.linalg.svd(two, full_matrices=False)
    two_values = singular**2 / (len(two) - 1)
    print("Unscaled prcomp DILG/DECI eigenvalues:", two_values)
    print(
        "Unscaled prcomp explained proportions:", two_values / two_values.sum()
    )
    print("Unscaled prcomp first five score rows (signs arbitrary):")
    print((two @ right.T)[:5])
    for component in range(2):
        print(
            f"Unscaled prcomp score variance {component + 1}: "
            f"{np.var(two @ right[component], ddof=1):.6f}"
        )
    harman = pd.read_csv(DATA / "ch02-Harman23-correlation.csv", index_col=0)
    h_values, h_loadings, _ = principal(harman, 2)
    _, h_rotated, h_weights = principal(harman, 2, rotation="varimax")
    assert np.allclose(h_loadings @ h_loadings.T, h_rotated @ h_rotated.T)
    retained = np.sum(h_loadings**2, axis=0)
    print("Harman23 correlation eigenvalues:", h_values)
    print("Harman proportions of all original variance:", retained / 8)
    print(
        "Harman proportions within retained variance:",
        retained / retained.sum(),
    )
    print(
        f"Harman retained fraction of original variance: {
            retained.sum() / 8:.8f}"
    )
    print("Harman unrotated loadings:")
    print(h_loadings)
    print("Harman Kaiser-normalized varimax loadings:")
    print(h_rotated)
    print("Harman rotated standardized-score weights, rounded as source:")
    print(np.round(h_weights, 2))
    _, source_h_cutoff, _ = parallel_pca(harman, 302)
    print(
        "Harman Gaussian parallel cutoff, source n.obs=302:", source_h_cutoff
    )
    _, h_cutoff, _ = parallel_pca(harman, 305)
    print("Harman Gaussian parallel cutoff, n.obs=305:", h_cutoff)
    _, h_cutoff_100, _ = parallel_pca(harman, 100)
    print(
        "Sensitivity if matrix n.obs omitted and default100 used:",
        h_cutoff_100,
    )
    reconstructed = standardized @ weights @ loadings.T
    error = np.sum((standardized - reconstructed) ** 2)
    expected = (len(raw) - 1) * values[1:].sum()
    assert np.allclose(error, expected)
    print(f"Rating rank-one reconstruction SSE: {error:.8f}")
    print(f"Discarded-eigenvalue SSE: {expected:.8f}")
    print(
        "Harman scores unavailable: individual rows not supplied."
    )
    plot_source_examples(
        judges, values, cutoff, boot, h_values, source_h_cutoff
    )
    print(
        "Checks passed: eigensystem, scores, rotation, reconstruction."
    )


if __name__ == "__main__":
    main()
