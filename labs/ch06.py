"""K-means descent and the lecture's exact source-data workflows.

The source workflows use the adjacent pure-Python Hartigan-Wong/R-RNG
port. --theory-only executes the independent mathematical construction.
No fitted R results are substituted for Python calculations.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import silhouette_score

DATA = Path(__file__).resolve().parent / "data"
ROOT = DATA.parents[1]


def standardize(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean(0)) / values.std(0, ddof=1)


def centered_scores(values):
    """Unscaled PCA coordinates; callers explicitly choose their geometry."""
    centered = values - np.mean(values, axis=0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * singular[:2], singular[:2] ** 2 / np.sum(singular**2)


def assigned_error(data, labels, centers):
    return np.sum((data - centers[labels]) ** 2)


def mathematical_checks():
    points = np.array([[5.0, 3.0], [-1.0, 1.0], [1.0, -2.0], [-3.0, -2.0]])
    labels = np.array([0, 0, 1, 1])
    centers = np.array([points[labels == j].mean(0) for j in range(2)])
    before = assigned_error(points, labels, centers)
    distances = cdist(points, centers, metric="sqeuclidean")
    assigned = distances.argmin(axis=1)
    reassigned = assigned_error(points, assigned, centers)
    updated = np.array([points[assigned == j].mean(0) for j in range(2)])
    after = assigned_error(points, assigned, updated)
    assert np.allclose([before, reassigned, after], [28, 27, 14])
    transfer_change = (2 / 3) * distances[1, 1] - 2 * distances[1, 0]
    assert np.isclose(after - before, transfer_change)
    assert np.array_equal(cdist(points, updated).argmin(1), assigned)
    print("Source four-point squared distances:\n", distances)
    print("Assignment/update WSS:", before, reassigned, after)
    print("Stable centers:\n", updated)
    print("Exact refitted transfer change:", transfer_change)
    # Exhaustive small-data verification: all nonempty two-group partitions.
    candidates = []
    for mask in range(1, 2 ** len(points) - 1):
        assignment = np.array([(mask >> i) & 1 for i in range(len(points))])
        means = np.array([points[assignment == j].mean(0) for j in range(2)])
        candidates.append(assigned_error(points, assignment, means))
    print("Exhaustive four-point optimal two-group WSS:", min(candidates))
    assert min(candidates) == 14
    mean = points.mean(0)
    trial = np.array([2.0, -3.0])
    left = np.sum((points - trial) ** 2)
    right = np.sum((points - mean) ** 2) + len(points) * np.sum(
        (trial - mean) ** 2
    )
    assert np.allclose(left, right)
    print("Mean-minimization identity, both sides:", left, right)
    example = np.array([[0.0], [2.0], [10.0], [12.0]])
    widths = silhouette_score(example, [0, 0, 1, 1])
    expected = np.mean([9 / 11, 7 / 9, 7 / 9, 9 / 11])
    assert np.allclose(widths, expected)
    print(f"Authored silhouette example mean: {widths:.8f}")
    print("Independent mathematical checks passed.")


def report_fit(name, result, columns):
    print("\n" + name)
    print("Sizes:", result["size"])
    print("Centers:")
    print(pd.DataFrame(result["centers"], columns=columns).to_string())
    print("Within-group sums of squares:", result["withinss"])
    print("Total WSS:", result["tot_withinss"])
    print("Total SS:", result["totss"])
    print("Between SS:", result["betweenss"])
    print("ifault:", result["ifault"])


def save_fit(name, result, index):
    pd.DataFrame({"cluster": result["cluster"]}, index=index).to_csv(
        DATA / f"ch06-{name}-membership.csv"
    )


def plot_source_workflow(
    utilities,
    utility_fit,
    wine,
    cultivar,
    wine_fit,
    raw_fractions,
    hartigan,
    wine_wss,
    arrests,
    arrest_fits,
    arrest_wss,
    silhouettes,
    wrapper_wss,
    wrapper_silhouettes,
    multi,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for j, center in enumerate(utility_fit["centers"]):
        axes[0].plot(range(8), center, "o-", label=f"Group {j + 1}")
    axes[0].set_xticks(range(8), utilities.columns, rotation=55, ha="right")
    axes[0].set(
        title="Utilities: all eight standardized features",
        ylabel="Standardized center coordinate",
    )
    axes[0].legend(ncol=2, fontsize=8)
    distances = cdist(utility_fit["centers"], utility_fit["centers"])
    image = axes[1].imshow(distances, cmap="magma")
    axes[1].set(
        title="Euclidean distances between fitted centers",
        xlabel="Group",
        ylabel="Group",
    )
    figure.colorbar(image, ax=axes[1])
    figure.savefig(ROOT / "assets/ch06-utilities.svg", metadata={"Date": None})
    plt.close(figure)

    scores, _ = centered_scores(wine)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    markers = ["o", "^", "s"]
    for group in range(1, 4):
        for actual in range(1, 4):
            subset = (wine_fit["cluster"] == group) & (cultivar == actual)
            axes[0].scatter(
                *scores[subset].T,
                marker=markers[actual - 1],
                color=f"C{group - 1}",
                s=18,
                alpha=0.8,
            )
    axes[0].set(
        title="Raw wine: color=cluster, shape=cultivar",
        xlabel=f"PC1 ({100 * raw_fractions[0]:.1f}%)",
        ylabel=f"PC2 ({100 * raw_fractions[1]:.1f}%)",
    )
    axes[1].plot(hartigan["Clusters"], hartigan["Hartigan"], "o-")
    axes[1].axhline(10, color="#fb7185", linestyle="--", label="Threshold10")
    axes[1].set(
        title="Literal FitKMeans: next-group count on x axis",
        xlabel="Candidate number of groups being added",
        ylabel="Hartigan statistic",
    )
    axes[1].legend()
    figure.savefig(ROOT / "assets/ch06-wine-raw.svg", metadata={"Date": None})
    plt.close(figure)

    figure, axes = plt.subplots(
        1, 3, figsize=(13, 3.8), constrained_layout=True
    )
    axes[0].plot(range(1, 16), wine_wss, "o-")
    axes[0].set(
        title="Standardized wine: source WSS loop",
        xlabel="k",
        ylabel="Within-group squared error",
    )
    axes[1].bar(list(multi["votes"]), list(multi["votes"].values()))
    axes[1].set(
        title="Literal NbClust26 votes",
        xlabel="Recommended k",
        ylabel="Number of criteria",
    )
    dindex = multi["raw_indices"]["Dindex"]
    axes[2].plot(dindex.index, dindex, "o-", label="Dindex")
    axes[2].set(title="NbClust Dindex", xlabel="Candidate k")
    figure.savefig(
        ROOT / "assets/ch06-wine-selection.svg", metadata={"Date": None}
    )
    plt.close(figure)

    scores, fractions = centered_scores(arrests)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, (count, fit) in zip(axes.flat, arrest_fits.items()):
        for group in range(1, count + 1):
            subset = scores[fit["cluster"] == group]
            axis.scatter(*subset.T, s=20, color=f"C{group - 1}")
            if len(subset) >= 3:
                hull = ConvexHull(subset)
                polygon = subset[hull.vertices]
                axis.fill(*polygon.T, alpha=0.12, color=f"C{group - 1}")
        axis.set(
            title=f"k={count}; WSS={fit['tot_withinss']:.3f}",
            xlabel=f"PC1 ({100 * fractions[0]:.1f}%)",
            ylabel=f"PC2 ({100 * fractions[1]:.1f}%)",
        )
    figure.savefig(
        ROOT / "assets/ch06-arrests-partitions.svg", metadata={"Date": None}
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    image = axes[0].imshow(squareform(pdist(arrests)), cmap="coolwarm")
    axes[0].set(
        title="Standardized USArrests distances",
        xlabel="Original state order",
        ylabel="Original state order",
    )
    figure.colorbar(image, ax=axes[0], shrink=0.7)
    axes[1].plot(range(1, 16), arrest_wss, "o-", label="Independent fits")
    axes[1].plot(range(1, 11), wrapper_wss, ":s", label="Wrapper: one start")
    axes[1].plot(
        range(1, 16),
        np.minimum.accumulate(arrest_wss),
        "--",
        label="Running minimum",
    )
    axes[1].set(title="WSS is distinct from transformed jump", xlabel="k")
    axes[1].legend(fontsize=8)
    axes[2].plot(range(2, 16), silhouettes, "o-", label="Custom: 25 starts")
    axes[2].plot(
        range(2, 11), wrapper_silhouettes, ":s", label="Wrapper: one start"
    )
    axes[2].set(title="Mean Euclidean silhouette", xlabel="k")
    axes[2].legend(fontsize=8)
    figure.savefig(
        ROOT / "assets/ch06-arrests-diagnostics.svg", metadata={"Date": None}
    )
    plt.close(figure)


def plot_state_coordinates(arrests_frame, arrests, initial, final):
    """Translate the source two-feature state labels and final PCA labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    scores, fractions = centered_scores(arrests)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for i, state in enumerate(arrests_frame.index):
        axes[0].text(
            arrests[i, 2],
            arrests[i, 0],
            state,
            fontsize=6,
            color=f"C{initial['cluster'][i] - 1}",
        )
        axes[1].text(
            *scores[i], state, fontsize=6, color=f"C{final['cluster'][i] - 1}"
        )
    for axis, values in zip(axes, [arrests[:, [2, 0]], scores]):
        span = np.ptp(values, axis=0)
        axis.set_xlim(
            values[:, 0].min() - 0.1 * span[0],
            values[:, 0].max() + 0.25 * span[0],
        )
        axis.set_ylim(
            values[:, 1].min() - 0.1 * span[1],
            values[:, 1].max() + 0.1 * span[1],
        )
    axes[0].set(
        title="Two-group labels in selected feature coordinates",
        xlabel="Standardized Urban Population",
        ylabel="Standardized Murder",
    )
    axes[1].set(
        title="Final source four-group state labels",
        xlabel=f"PC1 ({100 * fractions[0]:.1f}%)",
        ylabel=f"PC2 ({100 * fractions[1]:.1f}%)",
    )
    figure.savefig(
        ROOT / "assets/ch06-arrests-state-labels.svg", metadata={"Date": None}
    )
    plt.close(figure)


def run_source_workflow():
    from ch06_hartigan_wong import RRandom, kmeans_r
    from ch05_nbclust import nbclust_average

    utilities = pd.read_csv(DATA / "ch05-Utilities.csv", index_col=0)
    scaled_utilities = standardize(utilities)
    utility_fit = kmeans_r(scaled_utilities, 6, seed=2026)
    report_fit(
        "Utilities k6; controlled seed2026 (source seed unspecified)",
        utility_fit,
        utilities.columns,
    )
    print("Utilities memberships:")
    print(pd.Series(utility_fit["cluster"], index=utilities.index).to_string())
    print(
        "Utilities center distances:\n",
        cdist(utility_fit["centers"], utility_fit["centers"]),
    )
    save_fit("utilities", utility_fit, utilities.index)

    table = np.loadtxt(DATA / "ch06-wine-uci.data", delimiter=",")
    cultivar, wine = table[:, 0].astype(int), table[:, 1:]
    columns = [
        "Alcohol",
        "Malic_acid",
        "Ash",
        "Alcalinity",
        "Magnesium",
        "Phenols",
        "Flavanoids",
        "Nonflavonoid_phenols",
        "Proanthocyanins",
        "Color",
        "Hue",
        "OD280_OD315",
        "Proline",
    ]
    single = kmeans_r(wine, 3, seed=278613)
    multiple = kmeans_r(wine, 3, nstart=25, seed=278613)
    report_fit("Raw wine k3; one source start", single, columns)
    report_fit("Raw wine k3; 25 source starts", multiple, columns)
    print("Wine cultivar versus fitted cluster (labels need no alignment):")
    print(pd.crosstab(cultivar, multiple["cluster"]).to_string())
    _, raw_fractions = centered_scores(wine)
    print("Raw wine first two PCA display fractions:", raw_fractions)
    raw_wss = [
        kmeans_r(wine, k, nstart=25, seed=278613)["tot_withinss"]
        for k in range(1, 21)
    ]
    hartigan_values = [
        (len(wine) - k - 1) * (raw_wss[k - 1] / raw_wss[k] - 1)
        for k in range(1, 20)
    ]
    hartigan = pd.DataFrame(
        {"Clusters": range(2, 21), "Hartigan": hartigan_values}
    )
    hartigan["AddCluster"] = hartigan["Hartigan"] > 10
    print(
        "Literal useful::FitKMeans table:\n", hartigan.to_string(index=False)
    )
    hartigan.to_csv(DATA / "ch06-wine-hartigan.csv", index=False)

    scaled_wine = standardize(wine)
    # The source WSS function resets seed1234 at every k>=2.
    wine_wss = [np.sum(scaled_wine**2)] + [
        kmeans_r(scaled_wine, k, seed=1234)["tot_withinss"]
        for k in range(2, 16)
    ]
    print("Standardized wine one-start WSS:", np.array(wine_wss))
    multi = nbclust_average(
        wine, min_nc=2, max_nc=15, standardize=True, method="kmeans", seed=1234
    )
    print("Standardized wine NbClust26 recommendations:", multi["votes"])
    print("NbClust selected k:", multi["selected_k"])

    arrests_frame = pd.read_csv(DATA / "ch05-USArrests.csv", index_col=0)
    arrests = standardize(arrests_frame)
    rng = RRandom(123)
    arrest_fits = {
        k: kmeans_r(arrests, k, nstart=25, rng=rng) for k in range(2, 6)
    }
    print("USArrests k2..5 controlled seed123, sequential25starts.")
    print("The lecture's corresponding initial seed is not shown.")
    rng = RRandom(123)
    arrest_wss = [
        kmeans_r(arrests, k, nstart=10, rng=rng)["tot_withinss"]
        for k in range(1, 16)
    ]
    rng = RRandom(123)
    silhouettes = []
    for k in range(2, 16):
        fit = kmeans_r(arrests, k, nstart=25, rng=rng)
        silhouettes.append(silhouette_score(arrests, fit["cluster"]))
    print("USArrests WSS k1..15:", np.array(arrest_wss))
    print("USArrests mean silhouettes k2..15:", np.array(silhouettes))
    print("Silhouette-selected k:", np.argmax(silhouettes) + 2)
    # fviz_nbclust(...,kmeans) forwards no nstart: one-start default.
    # Controlled seeds make these separate wrapper demonstrations repeatable.
    rng = RRandom(123)
    wrapper_wss = [np.sum(arrests**2)] + [
        kmeans_r(arrests, k, rng=rng)["tot_withinss"] for k in range(2, 11)
    ]
    rng = RRandom(123)
    wrapper_silhouettes = [
        silhouette_score(arrests, kmeans_r(arrests, k, rng=rng)["cluster"])
        for k in range(2, 11)
    ]
    print("fviz_nbclust-equivalent WSS k1..10, one start:", wrapper_wss)
    print(
        "fviz_nbclust-equivalent silhouettes k2..10, one start:",
        wrapper_silhouettes,
    )
    final = kmeans_r(arrests, 4, nstart=25, seed=123)
    report_fit(
        "USArrests final source k4, seed123,nstart25",
        final,
        arrests_frame.columns,
    )
    print("Final state membership:")
    print(pd.Series(final["cluster"], index=arrests_frame.index).to_string())
    print("Raw-data means by final group:")
    print(arrests_frame.groupby(final["cluster"]).mean().to_string())
    save_fit("arrests-final", final, arrests_frame.index)
    assert np.isclose(final["tot_withinss"], 56.4031734582928)
    plot_source_workflow(
        utilities,
        utility_fit,
        wine,
        cultivar,
        multiple,
        raw_fractions,
        hartigan,
        wine_wss,
        arrests,
        arrest_fits,
        arrest_wss,
        silhouettes,
        wrapper_wss,
        wrapper_silhouettes,
        multi,
    )
    plot_state_coordinates(arrests_frame, arrests, arrest_fits[2], final)
    print("Source workflows executed with pure-Python Hartigan-Wong.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory-only", action="store_true")
    arguments = parser.parse_args()
    mathematical_checks()
    if not arguments.theory_only:
        run_source_workflow()
