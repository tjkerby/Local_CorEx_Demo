import numpy as np


def explore_cluster(data, indexes, num_vals=20):
    # Standardize variables (z-score)
    standardized_data = (data - data.mean()) / data.std(ddof=0)
    cluster = standardized_data.loc[indexes]
    cluster_means = cluster.mean()
    global_means = standardized_data.mean()
    cluster_diffs = cluster_means - global_means

    sorted_s = cluster_diffs.abs().sort_values(ascending=False)
    sorted_s_with_sign = cluster_diffs.loc[sorted_s.index]

    print(sorted_s_with_sign[:num_vals])

def n_largest_magnitude_indexes(arr, n, absolute=True):
    """
    Returns the indices of the n largest values (by magnitude) in the array.
    """
    if absolute:
        arr = np.abs(arr)
    sorted_indexes = np.argsort(arr)[::-1]
    return sorted_indexes[:n]

def sort_by_magnitude_reverse(tuple_list):
    sorted_list = sorted(tuple_list, key=lambda x: abs(x[0]), reverse=True)
    return sorted_list