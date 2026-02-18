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
        ax.imshow(np.mean(inputs[indexes[i]], axis=0).reshape(28, 28), cmap='gray')
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
    st.set_page_config(page_title="MNIST CorEx Analysis", layout="wide")
    st.title("🧠 MNIST CorEx Analysis & Delete Node Experiment")
    
    # Initialize session state for persistent plot
    if 'show_group_viz' not in st.session_state:
        st.session_state.show_group_viz = False
    if 'group_viz_fig' not in st.session_state:
        st.session_state.group_viz_fig = None
    
    # Load data
    with st.spinner("Loading models and data..."):
        data = load_data_and_models()
    
    st.success("✅ Models and data loaded!")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Tab 1: Group Visualization
    tab1, tab2, tab3 = st.tabs(["📊 Group Visualization", "🔧 Train CorEx", "📈 Delete Node Experiment"])
    
    with tab1:
        st.header("Average Pixel Values by Group")
        st.markdown("Showing the mean pixel values for each of the 20 groups.")
        
        if st.button("Generate Group Visualizations"):
            with st.spinner("Generating visualizations..."):
                st.session_state.group_viz_fig = plot_group_averages(data['inputs'], data['indexes'])
                st.session_state.show_group_viz = True
        
        if st.session_state.show_group_viz and st.session_state.group_viz_fig is not None:
            st.pyplot(st.session_state.group_viz_fig, use_container_width=True)
        else:
            st.info("Click the button above to generate group visualizations.")
        
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
        st.markdown("Select a group and hidden layer to train a CorEx model.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            group_id = st.selectbox("Select Group", range(20), key="group_select")
        
        with col2:
            layer_choice = st.selectbox(
                "Select Hidden Layer",
                options=[1, 2, 3],
                format_func=lambda x: f"Layer {x} (H{x})",
                key="layer_select"
            )
        
        # Map layer choice to layer index and config
        layer_map = {
            1: {'idx': 0, 'latent_dim': 500, 'encoder_layer': 1, 'encoder_idx': 1},
            2: {'idx': 1, 'latent_dim': 400, 'encoder_layer': 2, 'encoder_idx': 2},
            3: {'idx': 2, 'latent_dim': 300, 'encoder_layer': 3, 'encoder_idx': 3},
        }
        
        layer_config = layer_map[layer_choice]
        
        if st.button("Train CorEx Model", key="train_button"):
            with st.spinner(f"Training CorEx model for Group {group_id}, Layer {layer_choice}..."):
                try:
                    # Get data for this group and layer
                    group_indices = data['indexes'][group_id]
                    hidden_states = data['model_data'][layer_config['idx']][group_indices]
                    output_states = data['model_data'][4][group_indices]
                    
                    # Ensure data is float32 and properly formatted
                    hidden_states = np.asarray(hidden_states, dtype=np.float32)
                    output_states = np.asarray(output_states, dtype=np.float32)
                    x = np.concatenate([hidden_states, output_states], axis=1).astype(np.float32)
                    
                    # Train CorEx
                    corex_model = LinearCorex(30, seed=42, gaussianize='outliers')
                    corex_model.fit(x)
                    
                    st.session_state[f'corex_g{group_id}_h{layer_choice}'] = corex_model
                    st.session_state[f'corex_x_g{group_id}_h{layer_choice}'] = x
                    
                    # Display results
                    st.success(f"✅ Training completed!")
                    st.subheader("CorEx Total Correlation Score (TCS)")
                    tcs_value = corex_model.tcs
                    # Handle both scalar and array cases
                    if hasattr(tcs_value, '__len__'):
                        tcs_value = float(tcs_value[0]) if len(tcs_value) > 0 else 0.0
                    else:
                        tcs_value = float(tcs_value)
                    st.metric("TCS", f"{tcs_value:.4f}")
                    
                    # Show top factors by MI
                    st.subheader("Top Factors by Mutual Information")
                    try:
                        mi_scores = corex_model.moments['MI']
                        
                        # MI scores should be (30, 794) - aggregate across features
                        mi_scores = np.asarray(mi_scores)
                        
                        # If 2D, sum across features to get total MI per factor
                        if len(mi_scores.shape) == 2:
                            mi_per_factor = np.sum(mi_scores, axis=1)
                        else:
                            mi_per_factor = mi_scores.flatten()
                        
                        # Get top 5 factors
                        top_factor_indices = np.argsort(mi_per_factor)[-5:][::-1]
                        
                        mi_data = []
                        for factor_id in top_factor_indices:
                            mi_value = float(mi_per_factor[int(factor_id)])
                            mi_data.append({
                                'Factor': int(factor_id),
                                'Total MI': mi_value
                            })
                        
                        if mi_data:
                            mi_df = pd.DataFrame(mi_data)
                            st.dataframe(mi_df, use_container_width=True)
                        else:
                            st.info("No MI scores available.")
                    except Exception as e:
                        st.warning(f"Could not display MI scores: {str(e)}")
                    
                    st.info(f"💡 Model saved in session. Use the 'Delete Node Experiment' tab to run experiments on this model.")
                
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    with tab3:
        st.header("Delete Node Experiment")
        st.markdown("""
        Run the delete node experiment to measure how removing CorEx factors 
        affects classifier accuracy on the selected group.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            exp_group = st.selectbox("Select Group", range(20), key="exp_group_select")
        
        with col2:
            exp_layer = st.selectbox(
                "Select Hidden Layer",
                options=[1, 2, 3],
                format_func=lambda x: f"Layer {x}",
                key="exp_layer_select"
            )
        
        with col3:
            exp_factor = st.number_input("Select Factor", min_value=0, max_value=29, value=0, key="exp_factor")
        
        num_drop = st.slider("Number of Nodes to Drop", min_value=10, max_value=500, value=100, step=10)
        
        # Check if model exists
        model_key = f'corex_g{exp_group}_h{exp_layer}'
        
        if model_key in st.session_state:
            if st.button("Run Delete Node Experiment", key="run_experiment"):
                with st.spinner("Running delete node experiment..."):
                    try:
                        corex_model = st.session_state[model_key]
                        
                        # Map layer to hidden dim
                        layer_map_exp = {1: 500, 2: 400, 3: 300}
                        hidden_dim = layer_map_exp[exp_layer]
                        
                        # Get data for just this group (CorEx was trained on this group only)
                        group_idx = data['indexes'][exp_group]
                        group_inputs = data['inputs'][group_idx]
                        group_labels = data['labels'][group_idx]
                        group_indexes = [np.arange(len(group_idx))]  # Single group with all its samples
                        
                        # Run the delete node experiment
                        fig = plot_perturved_accuracy(
                            data['do_clf'],
                            corex_model,
                            group_inputs,
                            group_labels,
                            group_indexes,
                            factor_num=int(exp_factor),
                            hidden_layer_idx=exp_layer - 1,
                            num_clusters=1,
                            num_drop=num_drop,
                            hidden_dim=hidden_dim
                        )
                        
                        st.pyplot(fig, use_container_width=True)
                        st.success("✅ Experiment completed!")
                        
                    except Exception as e:
                        st.error(f"Error running experiment: {str(e)}")
        else:
            st.warning(f"⚠️ No CorEx model found for Group {exp_group}, Layer {exp_layer}.")
            st.info("Please train a CorEx model first using the 'Train CorEx' tab.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    MNIST CorEx Analysis Tool | Powered by Streamlit & PyTorch
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
