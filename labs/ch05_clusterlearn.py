"""Python port of clustlearn 1.0.0's bisecting k-means hierarchy.

The R package is by Eduardo Ruiz Sabajanes, Juan Jose Cuadrado Gallego,
and Universidad de Alcala, copyright 2023, MIT licensed. This adaptation
preserves the source split traversal, five-iteration option, tie rule,
empty-cluster safeguard, square-root total-SSE heights, and hclust layout.
Source: https://github.com/cran/clustlearn/tree/master/R

The Python random stream is explicit and differs from R's. An optional
initial-center sequence permits a reference check with identical starts;
the final reader and examples do not require R or private reference files.
"""
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, is_valid_linkage


def source_kmeans(data, initial_centers, max_iterations=5):
    """Lloyd updates with first-minimum ties and retained empty centers."""
    data = np.asarray(data, dtype=float)
    centers = np.asarray(initial_centers, dtype=float).copy()
    if max_iterations < 1 or len(centers) != 2:
        raise ValueError("Use two centers and positive max_iterations")
    for _ in range(max_iterations):
        old = centers.copy()
        distances = np.sum((old[:, None] - data[None, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=0)
        for index in range(2):
            if np.any(labels == index):
                centers[index] = data[labels == index].mean(axis=0)
        if np.array_equal(centers, old):
            break
    distances = np.sum((centers[:, None] - data[None, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=0)
    # The source changes the first label if one child is empty. Its saved
    # withinss is not recomputed, but that field never controls traversal.
    if np.unique(labels).size == 1:
        labels[0] = 1 - labels[0]
    return labels


def kmeanspp(data, rng):
    """Two-center D-squared sampling; Python and R draws are distinct."""
    probabilities = np.full(len(data), 1 / len(data))
    centers = []
    for _ in range(2):
        index = rng.choice(len(data), p=probabilities)
        centers.append(data[index].copy())
        distance = np.min(np.sum(
            (np.asarray(centers)[:, None] - data[None, :]) ** 2,
            axis=2), axis=0)
        total = distance.sum()
        probabilities = (distance / total if total > 0
                         else np.full(len(data), 1 / len(data)))
    return np.asarray(centers)


def divisive_clustering(data, max_iterations=5, seed=2026, starts=None):
    """Return source-style merge, heights, leaf order and SciPy linkage.

    Each observation starts in the root. Split depth-first, left before
    right, using source-style binary k-means. The height of a split is
    sqrt(SSE of its unsplit parent), not an inter-centroid distance.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("Need at least two finite feature rows")
    rng = np.random.default_rng(seed)
    replay = iter(starts) if starts is not None else None
    pending = [(np.arange(len(data)), len(data) - 1)]
    merges, heights, chosen = [], [], []
    while pending:
        indices, label = pending.pop(0)
        target = data[indices]
        centers = (np.asarray(next(replay)).reshape(2, data.shape[1])
                   if replay is not None else kmeanspp(target, rng))
        chosen.append(centers.copy())
        assignments = source_kmeans(target, centers, max_iterations)
        left = indices[assignments == 0]
        right = indices[assignments == 1]
        left_label = -int(left[0]) - 1 if len(left) == 1 else label - 1
        right_label = (-int(right[0]) - 1 if len(right) == 1
                       else label - len(left))
        merges.insert(0, [left_label, right_label])
        heights.insert(0, float(np.sum((target - target.mean(0)) ** 2)))
        if len(right) > 1:
            pending.insert(0, (right, right_label))
        if len(left) > 1:
            pending.insert(0, (left, left_label))
    merges = np.asarray(merges, dtype=int)
    heights = np.sqrt(heights)
    order = np.argsort(heights, kind="stable")
    inverse = np.argsort(order, kind="stable")
    positive = merges > 0
    merges[positive] = inverse[merges[positive] - 1] + 1
    merges = merges[order]
    heights = heights[order]
    leaves = list(merges[-1])
    while max(leaves) > 0:
        index = int(np.argmax(leaves))
        leaves[index:index + 1] = list(merges[leaves[index] - 1])
    leaf_order = -np.asarray(leaves, dtype=int)
    linkage = np.zeros((len(data) - 1, 4))
    sizes = np.ones(2 * len(data) - 1, dtype=int)
    for index, children in enumerate(merges):
        converted = [(-child - 1 if child < 0 else len(data) + child - 1)
                     for child in children]
        size = sizes[converted].sum()
        sizes[len(data) + index] = size
        linkage[index] = [*converted, heights[index], size]
    assert is_valid_linkage(linkage)
    assert sorted(leaf_order) == list(range(1, len(data) + 1))
    return {"merge": merges, "height": heights, "order": leaf_order,
            "linkage": linkage, "starts": chosen}


def main():
    """Run all six exact bundled datasets; no R dependency."""
    folder = Path(__file__).resolve().parent / "data"
    for number, count in enumerate((2, 2, 3, 3, 3, 3), 1):
        data = np.loadtxt(folder / f"ch05-clusterlearn-db{number}.csv",
                          delimiter=",", skiprows=1)
        result = divisive_clustering(data, seed=2026 + number)
        labels = fcluster(result["linkage"], count, criterion="maxclust")
        sizes = sorted(np.bincount(labels)[1:].tolist())
        print(f"db{number}: n={len(data)}; {len(result['merge'])} splits; "
              f"root height={result['height'][-1]:.8f}; "
              f"cut {count} sizes={sizes}")
    # Degenerate duplicate observations exercise the source empty-child fix.
    duplicate = divisive_clustering(np.zeros((4, 2)), seed=1)
    assert len(duplicate["merge"]) == 3
    print("Duplicate-row safeguard and all six linkage structures passed.")


if __name__ == "__main__":
    main()
