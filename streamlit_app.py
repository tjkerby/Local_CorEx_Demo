"""
Streamlit app for MNIST CorEx analysis and delete node experiments.
"""

import sys
from pathlib import Path
from collections import Counter
import pickle

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

# Setup paths
PROJECT_ROOT = Path(__file__).parent
PAPER_MNIST_ROOT = PROJECT_ROOT / "paper_mnist"

# Add project roots to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PAPER_MNIST_ROOT))
sys.path.insert(0, str(PAPER_MNIST_ROOT / "mnist_classifier" / "model"))

# Import local modules
sys.path.insert(0, str(PAPER_MNIST_ROOT / "mnist_classifier"))

from nn_utils import get_hidden_states, load_models
from nn_plotting import plot_perturved_accuracy
from data import MNISTDataModule
from autoencoder_config import conf as ae_conf_raw
from config import conf as clf_conf_raw

# Try to import local_corex, but provide fallbacks
try:
    from local_corex import LinearCorex, partition_data
    from local_corex.utils.plotting import hidden_state_plot, multi_rep_plot
    from nn_plotting import plot_logit_effects
    HAS_COREX = True
except ImportError:
    HAS_COREX = False
    st.warning("⚠️ local-corex library not installed. Some features will be limited.")


@st.cache_resource
def load_data_and_models():
    """Load models, data, and compute clusters."""
    # Ensure MNIST is downloaded
    MNIST(str(PROJECT_ROOT), train=False, download=True, transform=transforms.ToTensor())
    
    # Load configs
    ae_conf = ae_conf_raw['autoencoder']
    clf_conf = clf_conf_raw['classifier']
    
    # Setup paths
    base_path = PAPER_MNIST_ROOT / "mnist_classifier" / "model"
    ae_ckpt = str(base_path / 'mnist_ae_epoch=091-val_loss=0.5937.ckpt')
    clf_ckpt = str(base_path / 'mnist_clf_epoch=068-val_loss=0.0006.ckpt')
    
    # Load models
    do_ae, do_clf = load_models(ae_ckpt, clf_ckpt, ae_conf, clf_conf)
    
    # Load data
    data_module = MNISTDataModule(clf_conf, str(PROJECT_ROOT))
    data_module.setup('predict')
    
    # Get hidden states
    model_data = get_hidden_states(do_clf, data_module, device=do_clf.device, 
                                   num_layers=len(clf_conf['hidden_layers']))
    
    inputs = model_data[4]
    labels = model_data[5]
    
    # Load or compute clusters
    pickle_path = PAPER_MNIST_ROOT / "mnist_20_indexes.pkl"
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            indexes = pickle.load(f)
    else:
        indexes = partition_data(inputs, n_partitions=20, phate_dim=10, n_jobs=-2, seed=42)
        with open(pickle_path, 'wb') as f:
            pickle.dump(indexes, f)
    
    return {
        'do_ae': do_ae,
        'do_clf': do_clf,
        'data_module': data_module,
        'model_data': model_data,
        'inputs': inputs,
        'labels': labels,
        'indexes': indexes,
        'ae_conf': ae_conf,
        'clf_conf': clf_conf,
        'base_path': base_path
    }


def plot_group_averages(inputs, indexes):
    """Plot average pixel values for each group."""
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(np.mean(inputs[indexes[i]], axis=0).reshape(28, 28), cmap='viridis')
        ax.set_title(f'Group {i}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def compute_base_accuracies(do_clf, inputs, labels, indexes):
    """Compute baseline accuracies for each group."""
    do_clf.to('cpu')
    pred_digit = do_clf(torch.tensor(inputs, dtype=torch.float32)).max(1).indices.detach().numpy()
    
    accuracies = []
    for i in range(len(indexes)):
        acc = 100 * np.mean(pred_digit[indexes[i]] == labels[indexes[i]])
        accuracies.append(acc)
    
    return accuracies


def main():
    st.set_page_config(page_title="MNIST Classifier CorEx Analysis", layout="wide")
    st.title("🧠 MNIST Classifier CorEx Analysis")
    
    # Initialize session state for persistent plot and selected group
    if 'show_group_viz' not in st.session_state:
        st.session_state.show_group_viz = False
    if 'group_viz_fig' not in st.session_state:
        st.session_state.group_viz_fig = None
    if 'selected_group' not in st.session_state:
        st.session_state.selected_group = 0
    if 'selected_model_key' not in st.session_state:
        st.session_state.selected_model_key = None
    if 'last_delete_nodes_fig' not in st.session_state:
        st.session_state.last_delete_nodes_fig = None
    if 'last_delete_nodes_key' not in st.session_state:
        st.session_state.last_delete_nodes_key = None
    if 'mi_summary_by_model' not in st.session_state:
        st.session_state.mi_summary_by_model = {}

    def parse_model_key(model_key: str):
        try:
            group_str, layer_str = model_key.replace("corex_g", "").split("_h")
            return int(group_str), int(layer_str)
        except Exception:
            return None, None
    
    # Load data
    with st.spinner("Loading models and data..."):
        data = load_data_and_models()
    
    st.success("✅ Models and data loaded!")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    st.header("Group Visualization")
    st.markdown("Generate the full group visualizations from any tab.")

    if st.button("Generate Group Visualizations"):
        with st.spinner("Generating visualizations..."):
            st.session_state.group_viz_fig = plot_group_averages(data['inputs'], data['indexes'])
            st.session_state.show_group_viz = True

    if st.session_state.show_group_viz and st.session_state.group_viz_fig is not None:
        st.pyplot(st.session_state.group_viz_fig, use_container_width=True)
    else:
        st.info("Click the button above to generate group visualizations.")

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Choose a Group", "🔧 Train CorEx", "📈 Explore"])
    
    with tab1:
        st.header("Choose a Group")
        st.markdown("Select the group to use throughout the pipeline.")

        st.subheader("Select Group for Analysis")
        selected_group = st.selectbox(
            "Choose a group to analyze throughout the pipeline",
            range(20),
            index=st.session_state.selected_group,
            key="tab1_group_select"
        )

        # Update session state if changed
        if selected_group != st.session_state.selected_group:
            st.session_state.selected_group = selected_group

        st.info(f"📌 Group {st.session_state.selected_group} is selected for downstream analysis.")

        # Show selected group label distribution and average image
        group_indices = data['indexes'][st.session_state.selected_group]
        group_labels = data['labels'][group_indices]
        label_counts = Counter(group_labels)
        sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        counts_text = ", ".join([f"{label}:{count}" for label, count in sorted_counts])
        st.markdown(f"**Label distribution:** {counts_text}")

        avg_fig, avg_ax = plt.subplots(figsize=(2.5, 2.5))
        avg_ax.imshow(np.mean(data['inputs'][group_indices], axis=0).reshape(28, 28), cmap='viridis')
        avg_ax.set_title(f"Group {st.session_state.selected_group}", fontsize=10)
        avg_ax.axis('off')
        st.pyplot(avg_fig, use_container_width=False)

        st.divider()

        st.subheader("Group Statistics")
        
        # Show group statistics
        st.subheader("Group Statistics")
        group_stats = []
        for i, idx_group in enumerate(data['indexes']):
            group_samples = data['labels'][idx_group]
            digit_counts = Counter(group_samples)
            group_stats.append({
                'Group': i,
                'Size': len(group_samples),
                'Main Digit': max(digit_counts, key=digit_counts.get),
                'Main Digit Count': max(digit_counts.values())
            })
        
        stats_df = pd.DataFrame(group_stats)
        st.dataframe(stats_df, use_container_width=True)
    
    with tab2:
        st.header("Train CorEx Model")
        st.markdown(f"Training CorEx model for **Group {st.session_state.selected_group}**.")

        # Show trained models and allow selection
        trained_model_keys = sorted(
            [k for k in st.session_state.keys() if k.startswith("corex_g") and "_h" in k]
        )
        trained_model_labels = []
        for key in trained_model_keys:
            g, h = parse_model_key(key)
            if g is not None and h is not None:
                trained_model_labels.append((f"Group {g} | Layer {h}", key))

        if trained_model_labels:
            label_to_key = {label: key for label, key in trained_model_labels}
            default_label = trained_model_labels[0][0]
            if st.session_state.selected_model_key in trained_model_keys:
                for label, key in trained_model_labels:
                    if key == st.session_state.selected_model_key:
                        default_label = label
                        break

            selected_label = st.selectbox(
                "Select trained model for experiments",
                options=[label for label, _ in trained_model_labels],
                index=[label for label, _ in trained_model_labels].index(default_label),
                key="trained_model_select"
            )
            st.session_state.selected_model_key = label_to_key[selected_label]
            if st.session_state.selected_model_key in st.session_state.mi_summary_by_model:
                st.subheader("Top Factors by Majority-Class Logit MI")
                st.dataframe(
                    st.session_state.mi_summary_by_model[st.session_state.selected_model_key],
                    use_container_width=True
                )
        else:
            st.info("No trained models yet. Train a model below to enable experiments.")
        
        # Use the group from Tab 1
        group_id = st.session_state.selected_group
        
        # Display group info
        group_indices = data['indexes'][group_id]
        group_labels = data['labels'][group_indices]
        digit_counts = Counter(group_labels)
        st.info(f"📊 Group {group_id}: {len(group_labels)} samples | Main digit: {max(digit_counts, key=digit_counts.get)}")
        
        layer_choice = st.selectbox(
            "Select Hidden Layer",
            options=[1, 2, 3],
            format_func=lambda x: f"Layer {x} (H{x})",
            key="layer_select"
        )
        
        # Map layer choice to layer index and config
        layer_map = {
            1: {'idx': 1, 'latent_dim': 500, 'encoder_idx': 0},
            2: {'idx': 2, 'latent_dim': 400, 'encoder_idx': 1},
            3: {'idx': 3, 'latent_dim': 300, 'encoder_idx': 2},
        }
        
        layer_config = layer_map[layer_choice]
        
        if st.button("Train CorEx Model", key="train_button"):
            with st.spinner(f"Training CorEx model for Group {group_id}, Layer {layer_choice}..."):
                try:
                    # Get data for this group and layer
                    group_indices = data['indexes'][group_id]
                    hidden_states = data['model_data'][layer_config['idx']][group_indices]
                    output_states = data['model_data'][0][group_indices]
                    
                    # Ensure data is float32 and properly formatted
                    hidden_states = np.asarray(hidden_states, dtype=np.float32)
                    output_states = np.asarray(output_states, dtype=np.float32)
                    x = np.concatenate([hidden_states, output_states], axis=1).astype(np.float32)
                    
                    # Train CorEx
                    corex_model = LinearCorex(30, seed=42, gaussianize='outliers')
                    corex_model.fit(x)
                    
                    new_model_key = f'corex_g{group_id}_h{layer_choice}'
                    st.session_state[new_model_key] = corex_model
                    st.session_state[f'corex_x_g{group_id}_h{layer_choice}'] = x
                    st.session_state.selected_model_key = new_model_key
                    
                    # Display results
                    st.success(f"✅ Training completed!")
                    st.subheader("CorEx Total Correlation Score (TCS)")

                    st.metric("TCS", f"{np.sum(corex_model.tcs):.4f}")
                    
                    # Show top factors by MI with majority class logits
                    st.subheader("Top Factors by Majority-Class Logit MI")
                    try:
                        mi_scores = corex_model.moments['MI']
                        mi_scores = np.asarray(mi_scores)

                        majority_class = max(digit_counts, key=digit_counts.get)

                        if len(mi_scores.shape) == 2 and mi_scores.shape[1] >= 10:
                            # last 10 columns correspond to logits
                            logits_mi = mi_scores[:, -10:]
                            majority_mi = logits_mi[:, int(majority_class)]
                        else:
                            majority_mi = mi_scores.flatten()

                        # Get top 5 factors by majority-class MI
                        top_factor_indices = np.argsort(majority_mi)[-5:][::-1]

                        mi_data = []
                        for factor_id in top_factor_indices:
                            mi_value = float(majority_mi[int(factor_id)])
                            mi_data.append({
                                'Factor': int(factor_id),
                                f'MI w/ class {majority_class}': mi_value
                            })
                        
                        if mi_data:
                            mi_df = pd.DataFrame(mi_data)
                            st.dataframe(mi_df, use_container_width=True)
                            st.session_state.mi_summary_by_model[new_model_key] = mi_df
                        else:
                            st.info("No MI scores available.")
                    except Exception as e:
                        st.warning(f"Could not display MI scores: {str(e)}")
                    
                    st.info(f"💡 Model saved in session. Use the 'Delete Node Experiment' tab to run experiments on this model.")
                
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    with tab3:
        st.header("Experiments")
        st.markdown(
            f"Use **Group {st.session_state.selected_group}** as the analysis anchor. "
            "Run multiple experiments below without re-selecting the group."
        )

        # Use the selected model if available; otherwise fall back to selected group
        exp_group = st.session_state.selected_group
        exp_layer = 1
        if st.session_state.selected_model_key:
            sel_group, sel_layer = parse_model_key(st.session_state.selected_model_key)
            if sel_group is not None and sel_layer is not None:
                exp_group, exp_layer = sel_group, sel_layer

        st.info(f"📌 Using CorEx model trained on Group {exp_group}, Layer {exp_layer}")

        # Shared controls
        col1, col2 = st.columns(2)

        with col1:
            st.selectbox(
                "Selected Hidden Layer",
                options=[exp_layer],
                format_func=lambda x: f"Layer {x}",
                key="exp_layer_select",
                disabled=True
            )

        with col2:
            exp_factor = st.number_input(
                "Select Factor",
                min_value=0,
                max_value=29,
                value=0,
                key="exp_factor"
            )

        exp_tab1, exp_tab2, exp_tab3 = st.tabs([
            "🧪 Delete Nodes",
            "🧬 Perturb Reconstructions",
            "🧠 Logit Impacts"
        ])

        # Check if model exists
        model_key = f'corex_g{exp_group}_h{exp_layer}'

        with exp_tab1:
            st.subheader("Delete Nodes Experiment")
            st.markdown(
                "Measure how removing nodes associated with a CorEx factor affects accuracy across all groups."
            )
            num_drop = st.slider(
                "Number of Nodes to Drop",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )

            if model_key in st.session_state:
                if st.button("Run Delete Node Experiment", key="run_experiment"):
                    with st.spinner("Running delete node experiment..."):
                        try:
                            corex_model = st.session_state[model_key]

                            # Map layer to hidden dim
                            layer_map_exp = {1: 500, 2: 400, 3: 300}
                            hidden_dim = layer_map_exp[exp_layer]

                            # Run the delete node experiment on ALL data
                            fig, diff_probs = plot_perturved_accuracy(
                                data['do_clf'],
                                corex_model,
                                data['inputs'],
                                data['labels'],
                                data['indexes'],
                                factor_num=int(exp_factor),
                                hidden_layer_idx=exp_layer - 1,
                                num_clusters=20,
                                num_drop=num_drop,
                                hidden_dim=hidden_dim,
                                return_probs=True
                            )

                            diff_key = f"diff_probs_g{exp_group}_h{exp_layer}_f{int(exp_factor)}_n{num_drop}"
                            st.session_state[diff_key] = diff_probs
                            st.session_state.last_delete_nodes_fig = fig
                            st.session_state.last_delete_nodes_key = diff_key

                            st.pyplot(fig, use_container_width=True)
                            st.success("✅ Experiment completed!")

                        except Exception as e:
                            st.error(f"Error running experiment: {str(e)}")
                if st.session_state.last_delete_nodes_fig is not None:
                    st.subheader("Last Result")
                    st.pyplot(st.session_state.last_delete_nodes_fig, use_container_width=True)
            else:
                st.warning(f"⚠️ No CorEx model found for Group {exp_group}, Layer {exp_layer}.")
                st.info("Please train a CorEx model first using the 'Train CorEx' tab.")

        with exp_tab2:
            st.subheader("Perturb Reconstructions (Hidden State Plot)")
            st.markdown(
                "Visualize reconstructions for the selected factor using the trained CorEx model."
            )

            scaler = st.slider(
                "Reconstruction Scaling",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1
            )

            if model_key in st.session_state:
                try:
                    corex_model = st.session_state[model_key]

                    # Map layer to config for reconstruction
                    layer_map = {
                        1: {'idx': 1, 'latent_dim': 500, 'encoder_idx': 0},
                        2: {'idx': 2, 'latent_dim': 400, 'encoder_idx': 1},
                        3: {'idx': 3, 'latent_dim': 300, 'encoder_idx': 2},
                    }
                    layer_config = layer_map[exp_layer]

                    # Prepare data for hidden_state_plot
                    group_indices = data['indexes'][exp_group]
                    hidden_states = data['model_data'][layer_config['idx']][group_indices]
                    output_states = data['model_data'][0][group_indices]
                    hidden_states = np.asarray(hidden_states, dtype=np.float32)
                    output_states = np.asarray(output_states, dtype=np.float32)
                    x = np.concatenate([hidden_states, output_states], axis=1).astype(np.float32)

                    plot_result = hidden_state_plot(
                        x,
                        corex_model,
                        data['do_ae'],
                        factors=[int(exp_factor)],
                        latent_dim=layer_config['latent_dim'],
                        encoder_layer=layer_config['encoder_idx'] + 1,
                        scaler=float(scaler),
                        output_dim=10
                    )

                    if plot_result is not None:
                        st.pyplot(plot_result, use_container_width=True)
                    else:
                        st.pyplot(plt.gcf(), use_container_width=True)

                except Exception as e:
                    st.error(f"Error generating reconstructions: {str(e)}")
            else:
                st.warning(f"⚠️ No CorEx model found for Group {exp_group}, Layer {exp_layer}.")
                st.info("Please train a CorEx model first using the 'Train CorEx' tab.")

        with exp_tab3:
            st.subheader("Logit Impacts")
            st.markdown(
                "Explore how logits change after node deletion for a chosen partition."
            )

            partition = st.selectbox(
                "Select Partition",
                range(20),
                index=exp_group,
                key="logit_partition_select"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                bottom_vals = st.number_input(
                    "Bottom classes",
                    min_value=1,
                    max_value=9,
                    value=2,
                    step=1,
                    key="logit_bottom_vals"
                )
            with col_b:
                top_vals = st.number_input(
                    "Top classes",
                    min_value=1,
                    max_value=9,
                    value=2,
                    step=1,
                    key="logit_top_vals"
                )

            diff_key = f"diff_probs_g{exp_group}_h{exp_layer}_f{int(exp_factor)}_n{num_drop}"
            if diff_key in st.session_state:
                diff_probs = st.session_state[diff_key]
                if not torch.is_tensor(diff_probs):
                    diff_probs = torch.tensor(diff_probs)

                class_names = [str(i) for i in range(10)]
                ave_diff = torch.mean(diff_probs[data['indexes'][partition]], dim=0)
                plot_logit_effects(
                    ave_diff,
                    class_names,
                    bottom_vals=int(bottom_vals),
                    top_vals=int(top_vals)
                )
                st.pyplot(plt.gcf(), use_container_width=True)
            else:
                st.warning("⚠️ No diff_probs found for the current settings.")
                st.info("Run the Delete Nodes experiment first to generate diff_probs.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    MNIST CorEx Analysis Tool | Powered by Streamlit & PyTorch
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
