"""Transparent Python DIANA, including the source's mean-absolute scaling."""

import numpy as np


def mean_absolute_standardize(data):
    """Match cluster::diana(stand=TRUE), which does not use sample SD."""
    raw = np.asarray(data, dtype=float)
    centered = raw - np.nanmean(raw, axis=0)
    scale = np.nanmean(np.abs(centered), axis=0)
    scale = np.where(scale == 0, 1, scale)
    return centered / scale


def missing_distances(data, metric="euclidean"):
    """Available-coordinate distances, scaled by total/available d."""
    raw = np.asarray(data, dtype=float)
    rows, dimension = raw.shape
    result = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(i):
            valid = np.isfinite(raw[i]) & np.isfinite(raw[j])
            if not valid.any():
                raise ValueError("No comparable features for a pair")
            delta = raw[i, valid] - raw[j, valid]
            factor = dimension / valid.sum()
            if metric == "euclidean":
                value = np.sqrt(np.sum(delta**2) * factor)
            elif metric == "manhattan":
                value = np.sum(np.abs(delta)) * factor
            else:
                raise ValueError("DIANA supports euclidean or manhattan here")
            result[i, j] = result[j, i] = value
    return result


def splinter_split(distance, members):
    """Move maximal positive distance contrasts into a splinter."""
    party = list(members)
    means = {
        i: np.mean([distance[i, j] for j in party if i != j]) for i in party
    }
    seed = max(party, key=lambda i: (means[i], -i))
    splinter = [seed]
    party.remove(seed)
    while len(party) > 1:
        differences = {}
        for i in party:
            own = np.mean([distance[i, j] for j in party if i != j])
            other = np.mean(distance[i, splinter])
            differences[i] = own - other
        chosen = max(party, key=lambda i: (differences[i], -i))
        if differences[chosen] <= 0:
            break
        party.remove(chosen)
        splinter.append(chosen)
    return tuple(sorted(party)), tuple(sorted(splinter))


def diana(distance):
    """Return a SciPy linkage representation and a divisive coefficient.

    Equal comparisons use smallest original row index. Leaf order may differ
    from R even when cophenetic distances and the partition hierarchy agree.
    """
    matrix = np.asarray(distance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square dissimilarity matrix")
    if not np.allclose(matrix, matrix.T) or np.any(matrix < 0):
        raise ValueError("Dissimilarities must be nonnegative and symmetric")
    count = len(matrix)
    active = [tuple(range(count))]
    divisions = []
    singleton_parent_height = np.zeros(count)
    while any(len(c) > 1 for c in active):
        candidates = [c for c in active if len(c) > 1]
        cluster = max(
            candidates, key=lambda c: (np.max(matrix[np.ix_(c, c)]), -min(c))
        )
        height = float(np.max(matrix[np.ix_(cluster, cluster)]))
        first, second = splinter_split(matrix, cluster)
        active.remove(cluster)
        active.extend([first, second])
        divisions.append((cluster, first, second, height))
        for child in [first, second]:
            if len(child) == 1:
                singleton_parent_height[child[0]] = height
    identifiers = {frozenset([i]): i for i in range(count)}
    links = []
    for parent, first, second, height in reversed(divisions):
        left = identifiers[frozenset(first)]
        right = identifiers[frozenset(second)]
        links.append([left, right, height, len(parent)])
        identifiers[frozenset(parent)] = count + len(links) - 1
    maximum = np.max(matrix)
    coefficient = (
        1 - singleton_parent_height.mean() / maximum if maximum > 0 else 0.0
    )
    return np.array(links), coefficient


def diana_from_data(data, metric="euclidean", standardize=True):
    raw = mean_absolute_standardize(data) if standardize else np.asarray(data)
    distances = missing_distances(raw, metric)
    links, coefficient = diana(distances)
    return links, coefficient, distances
