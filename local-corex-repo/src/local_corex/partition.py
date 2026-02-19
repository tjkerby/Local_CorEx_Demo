"""Data partitioning utilities used by the Local CorEx workflow."""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
import phate
from sklearn.cluster import KMeans

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def partition_data(
    inputs: ArrayLike,
    n_partitions: int = 10,
    phate_dim: int = 10,
    n_jobs: int = -2,
    seed: int = 42,
    return_pred: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    """Partition data using PHATE + K-means as the local analysis backbone.

    Parameters
    ----------
    inputs : array-like of shape (n_samples, n_features)
        Input data to partition.
    n_partitions : int, default=10
        Number of clusters/partitions to generate.
    phate_dim : int, default=10
        Dimensionality of the PHATE embedding prior to clustering.
    n_jobs : int, default=-2
        How many CPUs PHATE should use. `-2` uses all but one core.
    seed : int, default=42
        Random seed for reproducibility.
    return_pred : bool, default=False
        When True, also return the raw cluster assignments for each sample.

    Returns
    -------
    partitions : list of ndarray
        Boolean masks (length ``n_partitions``) indicating which samples belong to each cluster.
    pred : ndarray, optional
        K-means assignments for every sample. Only returned when ``return_pred`` is True.
    """

    phate_operator = phate.PHATE(
        n_components=phate_dim,
        n_jobs=n_jobs,
        random_state=seed,
    )
    y_phate = phate_operator.fit_transform(inputs)

    kmeans = KMeans(
        n_clusters=n_partitions,
        random_state=seed,
        n_init="auto",
    )
    kmeans.fit(y_phate)
    pred = kmeans.predict(y_phate)

    partitions: List[np.ndarray] = []
    for cluster_idx in range(n_partitions):
        partitions.append(pred == cluster_idx)

    if return_pred:
        return partitions, pred
    return partitions
