"""Representation-independent splits, expressed as original trial indices."""
import numpy as np


def split_indices(y, ratios=(0.8, 0.2), strategy="stratified_chronological"):
    """Partition trials without shuffling; preserve order within each partition.

    Stratification preserves class proportions up to rounding, not signal
    distributions. Its class-specific time boundaries may overlap.
    """
    y = np.asarray(y)
    ratios = np.asarray(ratios, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if (ratios.ndim != 1 or len(ratios) < 2 or
            not np.all(np.isfinite(ratios)) or np.any(ratios <= 0) or
            not np.isclose(ratios.sum(), 1)):
        raise ValueError("ratios must contain at least two positive fractions summing to 1")
    if strategy == "stratified_chronological":
        groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    elif strategy == "chronological":
        groups = [np.arange(len(y))]
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    parts = [[] for _ in ratios]
    for indices in groups:
        boundaries = np.floor(np.cumsum(ratios)[:-1] * len(indices)).astype(int)
        for part, selected in zip(parts, np.split(indices, boundaries)):
            part.append(selected)
    return tuple(np.sort(np.concatenate(part)).astype(int) if part
                 else np.empty(0, dtype=int) for part in parts)


def split_data(X, y, ratio=0.8):
    """Compatibility helper: class-wise chronological train/test split."""
    X, y = np.asarray(X), np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must have equal trial counts")
    train, test = split_indices(y, (ratio, 1 - ratio))
    return X[train], y[train], X[test], y[test]
