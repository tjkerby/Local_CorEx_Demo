"""Neural network-specific plotting helpers for the MNIST experiments."""

from __future__ import annotations

import copy
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

try:
    from local_corex.base import LinearCorex
    from local_corex.utils import data as du
    HAS_COREX = True
except ImportError:
    HAS_COREX = False
    LinearCorex = None
    du = None

try:  # Prefer relative import when package context is available
    from . import nn_utils as nu
except ImportError:  # pragma: no cover - fallback when running notebook in-place
    import nn_utils as nu  # type: ignore


def _corex_component(corex_model: LinearCorex, factor: int) -> np.ndarray:
    moments = getattr(corex_model, "moments", None)
    if not moments or "MI" not in moments:
        raise ValueError("corex_model must be fit so that moments['MI'] is available.")
    components = moments["MI"]
    if factor < 0 or factor >= len(components):
        raise IndexError(f"factor index {factor} is out of bounds for {len(components)} components.")
    return components[factor]


def plot_perturved_accuracy(
    clf,
    corex_model: LinearCorex,
    inputs: np.ndarray,
    labels: Sequence[int],
    indexes: Sequence[Sequence[int]],
    base_probs=None,
    factor_num: int = 0,
    hidden_layer_idx: int = 0,
    num_clusters: int = 20,
    num_drop: int = 35,
    hidden_dim: int = 200,
    return_probs: bool = False,
):
    """Visualize accuracy deltas per cluster after pruning a hidden layer.
    
    - Keeps inference on CPU (for demo stability).
    - Optionally returns diff_probs = altered_probs - base_probs.
      If base_probs is None, it uses the baseline probs from `clf`.
    """
    title = f"H{hidden_layer_idx + 1}"

    nodes = du.n_largest_magnitude_indexes(
        _corex_component(corex_model, factor_num)[:hidden_dim], num_drop
    ).tolist()

    clf_clone = copy.deepcopy(clf)
    clf_clone = nu.prune_model_node(clf_clone, hidden_layer_idx, nodes)

    # Ensure models + inference are on CPU
    device = torch.device("cpu")
    clf = clf.to(device)
    clf_clone = clf_clone.to(device)

    x = torch.tensor(inputs, dtype=torch.float32, device=device)

    with torch.no_grad():
        baseline_logits = clf(x)
        baseline_probs = torch.nn.functional.softmax(baseline_logits, dim=1)
        baseline_pred = baseline_probs.max(1).indices.cpu().numpy()

        if base_probs is None:
            base_probs = baseline_probs
        else:
            # Ensure provided base_probs is a CPU tensor for consistent subtraction
            if not torch.is_tensor(base_probs):
                base_probs = torch.tensor(base_probs)
            base_probs = base_probs.to(device)

        altered_logits = clf_clone(x)
        altered_probs = torch.nn.functional.softmax(altered_logits, dim=1)
        altered_pred = altered_probs.max(1).indices.cpu().numpy()

    base_accuracies: List[float] = []
    altered_accuracies: List[float] = []
    for i in range(num_clusters):
        idx = indexes[i]
        base_accuracies.append(100 * np.mean(baseline_pred[idx] == labels[idx]))
        altered_accuracies.append(100 * np.mean(altered_pred[idx] == labels[idx]))

    diff = [np.round(altered_accuracies[i] - base_accuracies[i], 3) for i in range(num_clusters)]

    columns = ["Factor #", "# dropped"] + [str(i) for i in range(num_clusters)]
    values = [factor_num + 1, num_drop] + diff
    temp_df = pd.DataFrame([values], columns=columns)

    fig, ax = plt.subplots(figsize=(10, 7))
    temp_df.drop(["Factor #", "# dropped"], axis=1).plot(
        kind="bar",
        ax=ax,
        width=0.8,
        colormap="tab20",
        ylabel="% difference in accuracy from baseline",
        title=f"Dropping {num_drop} nodes from {title} based on LC factor {factor_num + 1}",
    )
    ax.set_xticks([])
    ax.title.set_fontsize(20)
    ax.yaxis.label.set_fontsize(16)
    plt.tick_params(labelsize=14)
    plt.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.225), prop={"size": 12})
    plt.close(fig)

    if return_probs:
        diff_probs = altered_probs - base_probs
        return fig, diff_probs

    return fig


def plot_perturved_accuracy_resnet(
    clf,
    corex_model: LinearCorex,
    val_loader,
    indexes: Sequence[Sequence[int]],
    device,
    base_accuracies: Sequence[float] | None = None,
    base_probs=None,
    factor_num: int = 0,
    num_drop: int = 35,
    hidden_dim: int = 512,
    return_probs: bool = True,
):
    """Plot accuracy deltas caused by pruning ResNet's FC layer nodes."""
    nodes = du.n_largest_magnitude_indexes(
        _corex_component(corex_model, factor_num)[:hidden_dim], num_drop
    ).tolist()

    clf_clone = copy.deepcopy(clf)
    clf_clone = nu.prune_resnet_fc_layer(clf_clone, nodes)

    if base_accuracies is None:
        base_accuracies, base_probs = nu.compute_cluster_accuracies(
            clf, val_loader, device, indexes, return_probs=True
        )
    altered_accuracies, alt_probs = nu.compute_cluster_accuracies(
        clf_clone, val_loader, device, indexes, return_probs=True
    )

    diff = [np.round(altered_accuracies[i] - base_accuracies[i], 3) for i in range(len(indexes))]

    fig, ax = plt.subplots(figsize=(10, 6))

    abs_diff = [abs(d) for d in diff]
    most_impacted_indices = sorted(range(len(abs_diff)), key=lambda i: abs_diff[i], reverse=True)[:5]
    impact_colors = ["red", "blue", "green", "purple", "orange"]

    colors = ["lightgray"] * len(diff)
    for rank, cluster_idx in enumerate(most_impacted_indices):
        colors[cluster_idx] = impact_colors[rank]

    ax.bar(range(len(diff)), diff, color=colors, width=0.8, edgecolor="none")

    for rank, cluster_idx in enumerate(most_impacted_indices):
        value = diff[cluster_idx]
        if abs(value) > 0.1:
            ax.text(
                cluster_idx,
                value - 1 if value < 0 else value + 0.2,
                str(cluster_idx),
                ha="center",
                va="top" if value < 0 else "bottom",
                fontsize=12,
            )

    ax.set_ylabel("% difference in accuracy from baseline", fontsize=16)
    ax.set_title(
        f"Dropping {num_drop} nodes based on LC factor {factor_num + 1}", fontsize=18, pad=20
    )
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=14)

    plt.tight_layout()
    plt.show()

    if return_probs:
        diff_probs = alt_probs - base_probs
        return diff_probs
    return None

def plot_logit_effects(ave_diff_prob, class_names, bottom_vals=3, top_vals=10):
    top_n_values, top_n_indices = torch.topk(ave_diff_prob, top_vals)
    bottom_n_values, bottom_n_indices = torch.topk(ave_diff_prob, bottom_vals, largest=False)
    top_n_values = top_n_values.__reversed__()
    top_n_indices = top_n_indices.__reversed__()
    values = np.concatenate((bottom_n_values, top_n_values))
    indices = np.concatenate((bottom_n_indices, top_n_indices))

    labels = [class_names[i] for i in indices]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=['red']*len(bottom_n_values) + ['blue']*len(top_n_values))
    ax.set_xlabel('Indices', fontsize=16)
    ax.set_ylabel('Values', fontsize=16)
    ax.set_title('Classes most affected by dropped nodes', fontsize=18)
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            # For positive values, place text above the bar
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points above
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=12)
        else:
            # For negative values, place text below the bar
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -3),  # 3 points below
                        textcoords='offset points',
                        ha='center', va='top', fontsize=12)
    
    # Add padding above the highest bar to prevent annotation cutoff
    y_min, y_max = ax.get_ylim()
    max_value = max(values)
    min_value = min(values)
    
    # Calculate padding as 10% of the total range, with a minimum padding
    range_val = max_value - min_value
    padding = max(0.02, range_val * 0.1)  # At least 0.02 padding or 10% of range
    
    # Extend limits with padding
    if max_value > 0:
        ax.set_ylim(top=max_value + padding)
    if min_value < 0:
        ax.set_ylim(bottom=min_value - padding)
    
    plt.axvline(x=bottom_vals-.5, color='black', linewidth=1.5)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


__all__ = [
    "plot_perturved_accuracy",
    "plot_perturved_accuracy_resnet",
    "plot_logit_effects"
]
