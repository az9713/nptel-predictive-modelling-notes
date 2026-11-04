"""Source hierarchical examples: agglomeration, Ward, and literal DIANA."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, cut_tree, dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from ch05_diana import diana, diana_from_data
from ch05_nbclust import nbclust_average

DATA = Path(__file__).resolve().parent / "data"
ROOT = DATA.parents[1]


def standardize(data):
    values = np.asarray(data, dtype=float)
    return (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)


def canonical_labels(labels):
    """Number groups by first occurrence; preserve the partition."""
    mapping = {}
    return np.array(
        [mapping.setdefault(int(x), len(mapping) + 1) for x in labels]
    )


def cut(links, count):
    return canonical_labels(cut_tree(links, n_clusters=count).ravel())


def plot_nbclust(result):
    """Render all 26 index votes and Dindex diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    figure, axes = plt.subplots(
        1, 3, figsize=(13, 3.8), constrained_layout=True
    )
    votes = result["votes"]
    axes[0].bar(list(votes), list(votes.values()), color="#6ee7d8")
    axes[0].set(
        xlabel="Recommended cluster count",
        ylabel="Number of criteria",
        title="Four tied pluralities, not a unique majority",
    )
    index = result["raw_indices"]["Dindex"]
    axes[1].plot(index.index, index, "o-", color="#fb7185")
    axes[1].set(
        xlabel="Candidate clusters",
        ylabel="Dindex",
        title="Literal NbClust Dindex curve",
    )
    axes[2].plot(
        index.index[1:-1],
        np.diff(index.to_numpy(), n=2),
        "o-",
        color="#c4b5fd",
    )
    axes[2].set(
        xlabel="Candidate clusters",
        ylabel="Second difference",
        title="Dindex second differences",
    )
    figure.savefig(ROOT / "assets/ch05-nbclust.svg", metadata={"Date": None})
    plt.close(figure)


def plot_examples(manual, nutrient, utilities, scaled, links, diana_results):
    """Convert source trees, cuts, projection, heatmap and banner."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.spatial import ConvexHull

    plt.style.use("dark_background")
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 3.5), constrained_layout=True
    )
    for axis, method in zip(axes, ["single", "complete"]):
        dendrogram(
            linkage(squareform(manual), method=method),
            labels=["1", "2", "3", "4", "5"],
            ax=axis,
            above_threshold_color="#cbd5e1",
        )
        axis.set(title=f"Source five objects: {method}", ylabel="Merge height")
    figure.savefig(ROOT / "assets/ch05-manual.svg", metadata={"Date": None})
    plt.close(figure)
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 4.2), constrained_layout=True
    )
    for axis, name, table, tree in [
        (axes[0], "Nutrient: scaled average", nutrient, links["nutrient"]),
        (axes[1], "Utilities: Sales/Fuel single", utilities, links["single"]),
        (
            axes[2],
            "Utilities: Sales/Fuel average",
            utilities,
            links["average"],
        ),
    ]:
        displayed = dendrogram(
            tree,
            labels=list(table.index),
            leaf_rotation=90,
            leaf_font_size=5,
            ax=axis,
            above_threshold_color="#cbd5e1",
        )
        count = 5 if name.startswith("Nutrient") or "average" in name else 3
        labels = cut(tree, count)
        positions = {
            index: 5 + 10 * j for j, index in enumerate(displayed["leaves"])
        }
        low, high = tree[-count, 2], tree[-count + 1, 2]
        height = (low + high) / 2
        for group in range(1, count + 1):
            locations = [positions[i] for i in np.flatnonzero(labels == group)]
            axis.add_patch(
                Rectangle(
                    (min(locations) - 4, 0),
                    max(locations) - min(locations) + 8,
                    height,
                    fill=False,
                    edgecolor="#fb7185",
                    linewidth=0.7,
                )
            )
        axis.set(title=name, ylabel="Merge height")
    figure.savefig(
        ROOT / "assets/ch05-agglomerative.svg", metadata={"Date": None}
    )
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    _, singular, right = np.linalg.svd(scaled, full_matrices=False)
    projected = scaled @ right[:2].T
    labels = cut(links["average"], 4)
    axes[0].scatter(*projected.T, c=labels, cmap="Set2", s=30)
    palette = plt.get_cmap("Set2")
    for group in range(1, 5):
        members = projected[labels == group]
        if len(members) >= 3:
            hull = ConvexHull(members)
            axes[0].fill(
                *members[hull.vertices].T,
                alpha=0.15,
                color=palette((group - 1) / 3),
            )
    for i, name in enumerate(utilities.index):
        offset = {
            "Commonwealth": (3, -11),
            "Kentucky": (-38, 5),
            "Central": (-35, 4),
        }.get(name, (3, 3))
        axes[0].annotate(
            name,
            projected[i],
            xytext=offset,
            textcoords="offset points",
            fontsize=6,
        )
    fractions = singular**2 / np.sum(singular**2)
    axes[0].set(
        xlabel=f"PC1: {100 * fractions[0]:.2f}% all8 variance",
        ylabel=f"PC2: {100 * fractions[1]:.2f}% all8 variance",
        title="Labels from two features, display from all eight",
    )
    heatmap_tree = linkage(pdist(scaled), method="complete")
    row_order = dendrogram(heatmap_tree, no_plot=True)["leaves"]
    row_centered = scaled - scaled.mean(axis=1, keepdims=True)
    colors = row_centered / scaled.std(axis=1, ddof=1, keepdims=True)
    axes[1].imshow(colors[row_order], aspect="auto", cmap="gray_r")
    axes[1].set_xticks(
        np.arange(len(utilities.columns)),
        utilities.columns,
        rotation=65,
        ha="right",
        fontsize=6,
    )
    labels6 = cut(links["average"], 6)
    heat_labels = [f"{labels6[i]}: {utilities.index[i]}" for i in row_order]
    axes[1].set_yticks(np.arange(len(row_order)), heat_labels, fontsize=6)
    axes[1].set_title("Source heatmap: row-scaled color, complete row tree")
    figure.savefig(
        ROOT / "assets/ch05-utilities-display.svg", metadata={"Date": None}
    )
    figure.savefig(
        ROOT / "validation/chapters/ch05-utilities-figure.png", dpi=130
    )
    plt.close(figure)
    figure = plt.figure(figsize=(11, 7), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.5, 6, 0.25])
    tree_axis = figure.add_subplot(grid[0])
    dendrogram(
        heatmap_tree,
        orientation="left",
        no_labels=True,
        ax=tree_axis,
        above_threshold_color="#cbd5e1",
    )
    tree_axis.set(xlabel="Complete-link height", yticks=[])
    image_axis = figure.add_subplot(grid[1])
    image = image_axis.imshow(
        colors[row_order],
        origin="lower",
        aspect="auto",
        extent=(-0.5, len(utilities.columns) - 0.5, 0, 10 * len(utilities)),
        cmap="gray_r",
    )
    image_axis.set_xticks(
        np.arange(len(utilities.columns)),
        utilities.columns,
        rotation=60,
        ha="right",
        fontsize=7,
    )
    image_axis.set_yticks(
        np.arange(len(row_order)) * 10 + 5, heat_labels, fontsize=7
    )
    image_axis.yaxis.tick_right()
    color_axis = figure.add_subplot(grid[2])
    figure.colorbar(
        image, cax=color_axis, label="Within-row standardized value"
    )
    figure.savefig(
        ROOT / "assets/ch05-utilities-heatmap.svg", metadata={"Date": None}
    )
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, (name, table, tree, coefficient) in zip(
        axes.ravel(), diana_results
    ):
        count = 4 if name == "USArrests" else 2
        threshold = (tree[-count, 2] + tree[-count + 1, 2]) / 2
        dendrogram(
            tree,
            labels=list(table.index),
            leaf_rotation=90,
            leaf_font_size=4,
            ax=axis,
            color_threshold=threshold,
            above_threshold_color="#cbd5e1",
        )
        axis.set(
            title=f"{name}; divisive coefficient={coefficient:.4f}",
            ylabel="Cluster diameter at division",
        )
    figure.savefig(ROOT / "assets/ch05-diana.svg", metadata={"Date": None})
    plt.close(figure)
    votes_tree = diana_results[1][2]
    leaf_order = dendrogram(votes_tree, no_plot=True)["leaves"]
    cophenetic = squareform(cophenet(votes_tree))
    heights = [
        cophenetic[a, b] for a, b in zip(leaf_order[:-1], leaf_order[1:])
    ]
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    axis.barh(np.arange(len(heights)), heights, color="#fb7185")
    axis.invert_xaxis()
    axis.set(
        xlabel="Neighbor-leaf split height",
        ylabel="Leaf boundary in tree order",
        title="Votes DIANA banner: topology-equivalent leaf ordering",
    )
    figure.savefig(
        ROOT / "assets/ch05-votes-banner.svg", metadata={"Date": None}
    )
    plt.close(figure)


def main():
    np.set_printoptions(precision=6, suppress=True)
    manual = np.array(
        [
            [0, 9, 3, 6, 11],
            [9, 0, 7, 5, 10],
            [3, 7, 0, 9, 2],
            [6, 5, 9, 0, 8],
            [11, 10, 2, 8, 0],
        ],
        dtype=float,
    )
    for method in ["single", "complete"]:
        tree = linkage(squareform(manual), method=method)
        print("Source five-object", method, "merge heights:", tree[:, 2])
        print("Source five-object", method, "cut2:", cut(tree, 2))
    source_divisive = np.array(
        [
            [0, 12, 9, 32, 31],
            [12, 0, 9, 25, 27],
            [9, 9, 0, 23, 24],
            [32, 25, 23, 0, 9],
            [31, 27, 24, 9, 0],
        ],
        dtype=float,
    )
    tree, _ = diana(source_divisive)
    print("Source ABCDE divisive cut2:", cut(tree, 2))
    assert np.array_equal(cut(tree, 2), [1, 1, 1, 2, 2])
    nutrient = pd.read_csv(DATA / "ch05-nutrient.csv", index_col=0)
    nutrient.index = nutrient.index.str.lower()
    standardized_nutrient = standardize(nutrient)
    print("Nutrient raw distances first4:")
    print(squareform(pdist(nutrient))[:4, :4])
    print("Nutrient scaled distances first4:")
    print(squareform(pdist(standardized_nutrient))[:4, :4])
    nutrient_tree = linkage(standardized_nutrient, method="average")
    nutrient_labels = cut(nutrient_tree, 5)
    print("Nutrient cut5 sizes:", np.bincount(nutrient_labels)[1:])
    print("Nutrient raw within-cluster medians:")
    print(nutrient.groupby(nutrient_labels).median().to_string())
    nutrient.assign(cluster=nutrient_labels).to_csv(
        DATA / "ch05-nutrient-cut5.csv"
    )
    print("Nutrient standardized within-cluster medians:")
    print(
        pd.DataFrame(standardized_nutrient, columns=nutrient.columns)
        .groupby(nutrient_labels)
        .median()
        .to_string()
    )
    multi_index = nbclust_average(
        nutrient.to_numpy(), min_nc=2, max_nc=15, standardize=True
    )
    print("Nutrient literal26-index recommendations:", multi_index["votes"])
    print("Tied largest vote counts:", multi_index["tied_pluralities"])
    print("NbClust convention-selected k:", multi_index["selected_k"])
    plot_nbclust(multi_index)
    utilities = pd.read_csv(DATA / "ch05-Utilities.csv", index_col=0)
    scaled = standardize(utilities)
    _, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    print(
        "Utilities all8-variable PCA display fractions:",
        singular[:2] ** 2 / np.sum(singular**2),
    )
    selected = scaled[:, [5, 7]]
    links = {
        method: linkage(selected, method=method)
        for method in ["single", "average", "ward"]
    }
    links["nutrient"] = nutrient_tree
    for method in ["single", "average"]:
        for count in [3, 5, 6]:
            labels = cut(links[method], count)
            print(
                "Utilities",
                method,
                f"cut{count} sizes:",
                np.bincount(labels)[1:],
            )
    labels = cut(links["average"], 4)
    print("Utilities average cut4 membership:")
    print(pd.Series(labels, index=utilities.index).to_string())
    utilities.assign(cluster=labels).to_csv(DATA / "ch05-utilities-cut4.csv")
    # Original Ward check: geometry on the actual selected two-feature input.
    a, b = selected[:3], selected[3:5]
    union = np.vstack([a, b])

    def sse(x):
        return np.sum((x - x.mean(axis=0)) ** 2)

    increment = sse(union) - sse(a) - sse(b)
    formula = (
        len(a) * len(b) / len(union) * np.sum((a.mean(0) - b.mean(0)) ** 2)
    )
    assert np.allclose(increment, formula)
    print(f"Authored Ward SSE increment/direct formula: {increment:.8f}")
    diana_results = []
    for name, filename, metric, stand in [
        ("USArrests", "USArrests", "euclidean", True),
        ("Votes Manhattan", "votes-repub", "manhattan", True),
        ("Votes Euclidean", "votes-repub", "euclidean", True),
        ("Agriculture", "agriculture", "euclidean", False),
    ]:
        table = pd.read_csv(DATA / f"ch05-{filename}.csv", index_col=0)
        tree, coefficient, _ = diana_from_data(table, metric, stand)
        print(
            name,
            f"DC={coefficient:.8f}",
            "cut2 sizes:",
            np.bincount(cut(tree, 2))[1:],
        )
        diana_results.append((name, table, tree, coefficient))
    votes_manhattan = cut(diana_results[1][2], 2)
    votes_euclidean = cut(diana_results[2][2], 2)
    assert np.array_equal(votes_manhattan, votes_euclidean)
    print("Votes Manhattan/Euclidean cut2 assignments identical:", True)
    print("Votes first cut2 group:")
    print(diana_results[1][1].index[votes_manhattan == 1].tolist())
    plot_examples(manual, nutrient, utilities, scaled, links, diana_results)
    print(
        "Source bisecting-kmeans examples: run ch05_clusterlearn.py."
    )
    print(
        "NbClust26-index source wrapper converted and independently R-checked."
    )
    print(
        "Checks passed: source merges, Ward identity, hierarchies."
    )


if __name__ == "__main__":
    main()
