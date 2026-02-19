import imageio
import numpy as np
import pandas as pd
import torch
import warnings
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
from scipy.spatial.distance import cosine
from torch import nn

from . import data as du


def _resolve_corex_model(corex_model, legacy_model, func_name):
    """Return whichever model is provided while supporting the legacy keyword."""
    if corex_model is not None:
        if legacy_model is not None and legacy_model is not corex_model:
            warnings.warn(
                f"{func_name} received both corex_model and deprecated latent_transformer; using corex_model.",
                UserWarning,
                stacklevel=3,
            )
        return corex_model
    if legacy_model is not None:
        warnings.warn(
            f"{func_name} argument 'latent_transformer' is deprecated; pass the model via 'corex_model'.",
            DeprecationWarning,
            stacklevel=3,
        )
        return legacy_model
    raise ValueError(f"{func_name} requires a fitted LinearCorex model.")


def _corex_component(corex_model, factor):
    """Fetch the MI vector for a latent factor from a fitted LinearCorex model."""
    moments = getattr(corex_model, "moments", None)
    if not moments or "MI" not in moments:
        raise ValueError("corex_model must be fit so that moments['MI'] is available.")
    components = moments["MI"]
    if factor < 0 or factor >= len(components):
        raise IndexError(f"factor index {factor} is out of bounds for {len(components)} components.")
    return components[factor]


def _corex_num_components(corex_model):
    """Infer the number of latent components from the model."""
    count = getattr(corex_model, "m", None)
    if count:
        return count
    moments = getattr(corex_model, "moments", None)
    if moments and "MI" in moments:
        return len(moments["MI"])
    raise ValueError("Unable to determine the number of components from the provided model.")


def plot_corex_vars(model, df, factor=0, n_vals=10, ax=None):
    '''
    Plots a barplot of the Mutual Information (MI) between the features in a dataframe and a 
    selected factor from the CorEx model.

    Args:
    - model: The CorEx model containing the mutual information (MI) values.
    - df: A pandas DataFrame containing the input data (columns representing features).
    - factor: The factor number (0-indexed) for which MI values should be plotted. Default is 0.
    - n_vals: The number of top features (with the highest MI) to display. Default is 10.
    - ax: Optional; a Matplotlib axis object to plot on. If None, a new plot will be created.

    Returns:
    - A horizontal bar plot showing the top `n_vals` features and their MI values with the 
      selected CorEx factor.

    Behavior:
    - The function extracts the top `n_vals` features with the largest MI (Mutual Information) 
      for a specified `factor` in the CorEx model.
    - If `ax` is provided, the plot is drawn on the given axis; otherwise, a new plot is created.
    - The feature names (from the DataFrame) and their corresponding MI values are displayed 
      as a horizontal bar plot.
    '''
    # Get the indexes of the top n_vals features with the highest MI for the specified factor.
    indexes = du.n_largest_magnitude_indexes(model.mis[factor], n_vals, absolute=False)[::-1]

    # Get the column names (feature names) and their corresponding MI values.
    top_features = df.columns[indexes]
    mi_values = model.mis[factor][indexes]

    # Create a new plot if no axis (ax) is provided.
    if ax is None:
        plt.xlabel(f"MI with Factor {factor + 1}")
        return plt.barh(top_features, mi_values)
    else:
        ax.set_xlabel(f'MI with Factor {factor + 1}')
        return ax.barh(top_features, mi_values)
    
def plot_n_factors(model, df, n=20, n_vals=10, cols=4):
    # Handle both integer and list inputs for n
    if isinstance(n, (list, tuple)):
        factors_to_plot = list(n)  # Convert to list if tuple
        num_plots = len(factors_to_plot)
    else:
        factors_to_plot = list(range(n))  # Create list [0, 1, 2, ..., n-1]
        num_plots = n
    
    # Calculate the number of rows needed based on the number of plots and columns
    rows = (num_plots + cols - 1) // cols  # Equivalent to ceil(num_plots / cols)
    
    # Create subplots with appropriate size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 1.1 + n_vals * .66), squeeze=False)
    
    # Flatten the axes array for easy indexing
    axes = axes.flatten()
    
    # Loop through each factor to plot
    for idx, factor in enumerate(factors_to_plot):
        plot_corex_vars(model, df, factor=factor, n_vals=n_vals, ax=axes[idx])
        axes[idx].set_title(f'Local CorEx Factor {factor + 1}')  # Optional: Set title for each subplot
    
    # Hide any remaining empty subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')  # Turn off axes for extra subplots

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def vis_diff_between_partitions(data, model_1, model_2, n_factors=5):
    for i in range(n_factors):
        max_dist = 0
        best_ind = None
        for j in range(len(model_2.mis)):
            cur_dist = 1 - cosine(model_1.mis[i], model_2.mis[j])
            if cur_dist > max_dist:
                max_dist = cur_dist
                best_ind = j
        fig = plt.figure(figsize=(12, 5), layout="constrained")
        spec = fig.add_gridspec(2, 2)
        ax0 = fig.add_subplot(spec[0, :])
        ax10 = fig.add_subplot(spec[1, 0])
        ax11 = fig.add_subplot(spec[1, 1])
        ax0.imshow(np.vstack([model_1.mis[i], model_2.mis[best_ind]]))
        ax0.set_title("Comparing factors M.I. across variables")
        plot_corex_vars(model_1, data, factor=i, n_vals=10, ax=ax10)
        ax10.set_title("Plotting top 10 values for first model")
        plot_corex_vars(model_2, data, factor=best_ind, n_vals=10, ax=ax11) 
        ax11.set_title("Plotting top 10 values for second model")
        plt.tight_layout()
        plt.show()

def plot_reconstructions(x, x_hat, num_plots=3):
    fig, axes = plt.subplots(num_plots,2, figsize=(9,num_plots*4))
    for i in range(num_plots):
        axes[i,0].imshow(x[i].reshape(28,28))
        axes[i,1].imshow(x_hat[i].reshape(28,28))
    plt.show()
    
def hidden_state_plot(
    x,
    corex_model=None,
    ae_model=None,
    factors=[0, 1, 2],
    scaler=1,
    latent_dim=200,
    encoder_layer=None,
    output_dim=10,
    input_dim=(28, 28),
    **kwargs,
):
    legacy_model = kwargs.pop("latent_transformer", None)
    if kwargs:
        unexpected = ", ".join(kwargs)
        raise TypeError(f"hidden_state_plot() got unexpected keyword arguments: {unexpected}")
    model = _resolve_corex_model(corex_model, legacy_model, "hidden_state_plot")
    if ae_model is None:
        raise ValueError("hidden_state_plot requires an autoencoder model (ae_model).")
    # Force all computation on CPU.
    device = torch.device("cpu")
    ae_model.to(device)
    ae_model.eval()
    
    x_mean = np.mean(x, axis=0)
    x_mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Determine remaining encoder if we want to perturb an intermediate layer.
    num_blocks = len(ae_model.conf["encoder_layers"])
    if encoder_layer is None or encoder_layer >= num_blocks:
        remaining_encoder = nn.Identity()
    else:
        block_size = 3 + int(ae_model.use_bn)
        block_end_index = encoder_layer * block_size
        encoder_list = list(ae_model.encoder)
        remaining_encoder = nn.Sequential(*encoder_list[block_end_index:])
    
    # Get final latent representation from autoencoder's encoder if needed.
    projected_bottleneck = remaining_encoder(x_mean_tensor[:, :latent_dim])
    
    for i in factors:
        perturbation = scaler * _corex_component(model, i)
        latent_p = x_mean_tensor + perturbation
        latent_m = x_mean_tensor - perturbation
        latent_tensor_p = torch.as_tensor(latent_p, dtype=torch.float32, device=device)
        latent_tensor_m = torch.as_tensor(latent_m, dtype=torch.float32, device=device)
        
        # If an intermediate encoder is used, pass the perturbed vectors through it.
        latent_p_final = remaining_encoder(latent_tensor_p[:, :latent_dim])
        latent_m_final = remaining_encoder(latent_tensor_m[:, :latent_dim])
        
        # Decode the final latent vectors.
        recon_center = ae_model.decoder(projected_bottleneck).detach().cpu().numpy().reshape(input_dim)
        recon_p = ae_model.decoder(latent_p_final).detach().cpu().numpy().reshape(input_dim)
        recon_m = ae_model.decoder(latent_m_final).detach().cpu().numpy().reshape(input_dim)
        
        # Plot the reconstructions and extra perturbation information.
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        images = [recon_m, recon_center, recon_p]
        clim_min = min(np.min(img) for img in images)
        clim_max = max(np.max(img) for img in images)
        
        for j, img in enumerate(images):
            axes[j].imshow(img, cmap='viridis', clim=(clim_min, clim_max))
            axes[j].set_title(f"Plot {j+1}")
        
        # For the extra plot, show part of the perturbation vector.
        extra = (perturbation if isinstance(perturbation, np.ndarray) 
                 else np.array(perturbation))[latent_dim:latent_dim+output_dim].reshape(1, output_dim)
        axes[3].imshow(extra, cmap='viridis')
        axes[3].set_title(f"Factor {i+1}")
        
        plt.show()
        
def multi_rep_plot(lc_model, num_latent_factors, dims=[(28,28),(10,10)], num_per_row=2):

    fig, axes = plt.subplots(int(np.ceil(num_latent_factors / num_per_row)), num_per_row*2, figsize=(6*num_per_row, 3*(num_latent_factors/num_per_row)))

    for i in range(num_latent_factors):
        row_index = i // (num_per_row)
        col_index = (i % num_per_row) * 2

        for j in range(2):  # Plot for two representations (input and first hidden layer) for each latent factor
            index = i * num_per_row + j  # Calculate index for moments array based on latent factor and representation

            ax = axes[row_index, col_index + j]
            if j == 0:
                ax.imshow(lc_model.moments['MI'][i][:(dims[0][0]*dims[0][1])].reshape(dims[0]))
            else:
                ax.imshow(lc_model.moments['MI'][i][(dims[0][0]*dims[0][1]):].reshape(dims[1]))
            ax.set_title(f"Latent Factor {i+1}, Rep {j+1}")

    plt.tight_layout()
    plt.show()
    
    
def convert_to_uint8(frame):
    """
    Converts a float image array (which might have negative values)
    to a uint8 image scaled between 0 and 255.
    """
    # Compute min and max
    mn = frame.min()
    mx = frame.max()
    if mx - mn == 0:
        # Avoid division by zero; return a zeros array.
        norm = np.zeros_like(frame)
    else:
        norm = (frame - mn) / (mx - mn)
    return (norm * 255).astype(np.uint8)

def concatenate_frames(original_frames, diff_frames, hist_frames, axis=1):
    """
    Concatenates each original frame with its corresponding difference frame.
    """
    concatenated = []
    for orig, diff, hist in zip(original_frames, diff_frames, hist_frames):
        concat_frame = np.concatenate([orig, diff, hist], axis=axis)
        # Convert to uint8 if needed
        if concat_frame.dtype != np.uint8:
            concat_frame = convert_to_uint8(concat_frame)
        concatenated.append(concat_frame)
    return concatenated

def compute_diff_frames(frames):
    """
    Computes difference frames for a list of frames.
    The first difference frame is set to zeros.
    The function precomputes the global maximum absolute difference across all frames
    and uses that to fix the normalization, ensuring consistency across frames.
    
    Args:
        frames: List of NumPy arrays representing the images.
    
    Returns:
        diff_frames: List of images (NumPy arrays) of the plotted differences.
    """
    # First, compute raw difference arrays.
    raw_diffs = []
    prev = None
    for frame in frames:
        if prev is None:
            diff = np.zeros_like(frame, dtype=np.float32)
        else:
            diff = frame.astype(np.float32) - prev.astype(np.float32)
        raw_diffs.append(diff)
        prev = frame
    
    # Compute the global maximum absolute difference.
    global_max = max(np.max(np.abs(diff)) for diff in raw_diffs)
    if global_max == 0:
        global_max = 1.0  # Avoid division by zero.
    
    # Create a TwoSlopeNorm that is fixed for all frames.
    norm = TwoSlopeNorm(vmin=-global_max, vcenter=0, vmax=global_max)
    
    # Now, plot each difference frame using the same normalization.
    diff_frames = []
    for diff in raw_diffs:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(diff, cmap='coolwarm', norm=norm)
        ax.axis('off')
        ax.set_title("Difference", fontsize=12)
        fig.tight_layout(pad=0)
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype='uint8').reshape((height, width, 4))
        image = image[:, :, :3]  # Discard the alpha channel
        
        diff_frames.append(image)
        plt.close(fig)
    
    return diff_frames

def get_hist_plots(z, perturbed_vals):
    plots = []
    for val in perturbed_vals:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(z[:, 0], bins=30, density=True, edgecolor='black')
        ax.axvline(x=val, color='red', linestyle='--', linewidth=2)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        # Add labels and title
        ax.set_xlim(left=perturbed_vals[0], right=perturbed_vals[-1])
        ax.set_xlabel('Component vals')
        ax.set_ylabel('Percentage')
        ax.set_title('Histogram of the Component')
        fig.tight_layout()

        # Convert plot to image array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype='uint8').reshape((height, width, 4))
        image = image[:, :, :3]  # Discard the alpha channel
        plots.append(image)
        plt.close(fig)  # Close the figure to free memory
    return plots


def create_perturbation_frames_exact(
    x,
    corex_model=None,
    ae_model=None,
    factor=0,
    scaler_increment=0.5,
    max_attempt=10,
    latent_dim=200,
    encoder_layer=None,
    input_dim=(28, 28),
    verbose=False,
    **kwargs,
):
    """
    Creates a GIF of reconstructions from perturbed latent representations.
    
    The function starts with a small perturbation and increments it until the decoder fails 
    (i.e. the perturbation "breaks"). It then builds a list of safe perturbation scalers (both 
    negative and positive) and creates a GIF showing the reconstructions at these different levels.
    
    Args:
        x: Input data sample (e.g. a flattened image).
        corex_model: A fitted LinearCorex model exposing ``inverse_transform`` and ``moments['MI']``.
        ae_model: The autoencoder model.
        factor: Which latent factor (component index) to perturb.
        scaler_increment: How much to increase the perturbation at each step.
        max_attempt: Maximum number of increments to try.
        latent_dim: The dimension of the latent space.
        encoder_layer: (Optional) If specified, uses an intermediate representation.
        output_dim: Extra dimension from the component vector (used only for display).
        input_dim: Tuple representing the shape of the input image.
        gif_path: File path to save the GIF.
        fps: Frames per second for the GIF.
    """
    legacy_model = kwargs.pop("latent_transformer", None)
    if kwargs:
        unexpected = ", ".join(kwargs)
        raise TypeError(f"create_perturbation_frames_exact() got unexpected keyword arguments: {unexpected}")
    model = _resolve_corex_model(corex_model, legacy_model, "create_perturbation_frames_exact")
    if ae_model is None:
        raise ValueError("create_perturbation_frames_exact requires an autoencoder model (ae_model).")

    device = torch.device("cpu")
    ae_model.to(device)
    ae_model.eval()
    
    x_mean = np.mean(x, axis=0)
    x_mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Determine remaining encoder (if using an intermediate encoder layer)
    num_blocks = len(ae_model.conf["encoder_layers"])
    if encoder_layer is None or encoder_layer >= num_blocks:
        remaining_encoder = nn.Identity()
    else:
        block_size = 3 + int(ae_model.use_bn)  # e.g. [Linear, Dropout, ReLU] plus BN if used.
        block_end_index = encoder_layer * block_size
        encoder_list = list(ae_model.encoder)
        remaining_encoder = nn.Sequential(*encoder_list[block_end_index:])
    
    # Determine the maximum safe perturbation scale.
    safe_scalers = []
    for k in range(1, max_attempt + 1):
        current_scaler = k * scaler_increment
        try:
            perturbation = current_scaler * _corex_component(model, factor)
            latent_p = x_mean_tensor + torch.as_tensor(perturbation, dtype=torch.float32, device=device)
            latent_m = x_mean_tensor - torch.as_tensor(perturbation, dtype=torch.float32, device=device)
            _ = ae_model.decoder(remaining_encoder(latent_p[:, :latent_dim]))
            _ = ae_model.decoder(remaining_encoder(latent_m[:, :latent_dim]))
        except Exception as e:
            if verbose:
                print(e)
            break
        safe_scalers.append(current_scaler)
    
    if len(safe_scalers) == 0:
        print("No safe perturbation found; check your corex_model or autoencoder.")
        return
    
    # Build list of scaler values: negatives, then 0, then positives.
    positive_scalers = safe_scalers 
    negative_scalers = [-s for s in reversed(positive_scalers)]
    all_scalers = negative_scalers + [0] + positive_scalers

    frames = []
    data = []
    for scaler_val in all_scalers:
        perturbation = scaler_val * _corex_component(model, factor)
        latent_perturbed = x_mean_tensor + torch.as_tensor(perturbation, dtype=torch.float32, device=device)
        latent_final = remaining_encoder(latent_perturbed[:, :latent_dim])
        recon = ae_model.decoder(latent_final).detach().cpu().numpy().reshape(input_dim)
        data.append(recon)
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(recon, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Scaler: {scaler_val:.2f}", fontsize=12)
        fig.tight_layout()
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype='uint8').reshape((height, width, 4))
        image = image[:, :, :3]  # Discard the alpha channel
        frames.append(image)
        plt.close(fig)
    return frames, data   

def create_perturbation_frames_project(
    perturbed_vals,
    corex_model=None,
    ae_model=None,
    factor=0,
    latent_dim=200,
    encoder_layer=None,
    input_dim=(28, 28),
    **kwargs,
):
    legacy_model = kwargs.pop("latent_transformer", None)
    if kwargs:
        unexpected = ", ".join(kwargs)
        raise TypeError(f"create_perturbation_frames_project() got unexpected keyword arguments: {unexpected}")
    model = _resolve_corex_model(corex_model, legacy_model, "create_perturbation_frames_project")
    if ae_model is None:
        raise ValueError("create_perturbation_frames_project requires an autoencoder model (ae_model).")
    device = torch.device("cpu")
    ae_model.to(device)
    ae_model.eval()
    
    # Determine remaining encoder (if using an intermediate encoder layer)
    num_blocks = len(ae_model.conf["encoder_layers"])
    if encoder_layer is None or encoder_layer >= num_blocks:
        remaining_encoder = nn.Identity()
    else:
        block_size = 3 + int(ae_model.use_bn)  # e.g. [Linear, Dropout, ReLU] plus BN if used.
        block_end_index = encoder_layer * block_size
        encoder_list = list(ae_model.encoder)
        remaining_encoder = nn.Sequential(*encoder_list[block_end_index:])
    
    ave_latent_rep = np.zeros(_corex_num_components(model))
    
    frames = []
    data = []
    for scaler_val in perturbed_vals:
        ave_latent_rep[factor] = scaler_val
        ave_projected_rep = model.inverse_transform(ave_latent_rep)
        latent_perturbed = torch.as_tensor(ave_projected_rep, dtype=torch.float32, device=device).unsqueeze(0)
        latent_final = remaining_encoder(latent_perturbed[:, :latent_dim])
        recon = ae_model.decoder(latent_final).detach().cpu().numpy().reshape(input_dim)
        data.append(recon)
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(recon, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Perturbed Val: {scaler_val:.2f}", fontsize=12)
        fig.tight_layout()
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype='uint8').reshape((height, width, 4))
        image = image[:, :, :3]  # Discard the alpha channel
        frames.append(image)
        plt.close(fig)
    return frames, data

def create_gif_with_differences(frames, data, hist_frames, gif_path='output.gif', fps=2, concat_axis=1):
    """
    Given a list of frames, computes difference frames, concatenates each frame with its difference,
    converts them to uint8, and saves as a GIF.
    """
    diff_frames = compute_diff_frames(data)
    combined_frames = concatenate_frames(frames, diff_frames, hist_frames, axis=concat_axis)
    
    # Ensure each frame is uint8
    uint8_frames = [frame if frame.dtype==np.uint8 else convert_to_uint8(frame) for frame in combined_frames]
    imageio.mimsave(gif_path, uint8_frames, fps=fps)
    print(f"GIF saved to {gif_path}")

def create_animation_with_differences(frames, data, hist_frames, fps=2, concat_axis=1):
    """
    Given lists of frames (from the perturbation images), corresponding raw data (to compute difference frames),
    and histogram frames, this function computes the difference frames, concatenates each corresponding frame,
    ensures the resulting frame is uint8, and returns an animation object.
    """
    diff_frames = compute_diff_frames(data)
    combined_frames = concatenate_frames(frames, diff_frames, hist_frames, axis=concat_axis)
    combined_frames = [frame if frame.dtype == np.uint8 else convert_to_uint8(frame)
                       for frame in combined_frames]
    
    height, width = combined_frames[0].shape[:2]
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.axis('off')
    
    im = ax.imshow(combined_frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    plt.close()
    ani = FuncAnimation(
        fig, update, frames=combined_frames, interval=1000/fps, blit=True, repeat=True
    )
    
    return ani

def generate_sim_matrix(model_1, model_2, n_components_1=None, n_components_2=None, model_1_title='Model_1', model_2_title='Model_2'):
    """
    Generates and visualizes a similarity (correlation) matrix between the factors of two models.

    Args:
        model_1: The first model, expected to have a 'mis' attribute (list/array of factor vectors).
        model_2: The second model, expected to have a 'mis' attribute.
        n_components_1: Number of factors to use from model_1 (default: all).
        n_components_2: Number of factors to use from model_2 (default: all).
        model_1_title: Title label for model_1 factors (used in axis label).
        model_2_title: Title label for model_2 factors (used in axis label).

    Behavior:
        - Computes the correlation between each pair of factors from model_1 and model_2.
        - Displays a heatmap of the resulting similarity matrix.

    Returns:
        None (shows a plot).
    """
    # Use all factors if n_components not specified
    if n_components_1 is None:
        n_components_1 = len(model_1.mis)
    if n_components_2 is None:
        n_components_2 = len(model_2.mis)

    factors_1 = np.array([model_1.mis[i] for i in range(n_components_1)])
    factors_2 = np.array([model_2.mis[i] for i in range(n_components_2)])

    def factor_similarity(a, b):
        return np.corrcoef(a.flatten(), b.flatten())[0, 1]

    similarity_matrix = np.zeros((n_components_1, n_components_2))
    for i in range(n_components_1):
        for j in range(n_components_2):
            similarity_matrix[i, j] = factor_similarity(factors_1[i], factors_2[j])

    plt.figure(figsize=(6, 5))
    plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(label='Correlation')
    plt.xlabel(f'{model_2_title} factor index')
    plt.ylabel(f'{model_1_title} factor index')
    plt.title('Factor Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_corex_mis_per_component(model, n_components=16, n_columns=4, component_shape=(28, 28), cbar_max=None):
    """
    Visualizes the mutual information scores (MIS) for components in a Corex model.

    Parameters:
    -----------
    model : local_corex.LinearCorex
        The trained Corex model containing mutual information scores.
    n_components : int, default=16
        The number of components to visualize.
    n_columns : int, default=4
        The number of columns in the grid layout.
    component_shape : tuple, default=(28, 28)
        The shape to reshape each component to (e.g., for MNIST, 28x28).
    cbar_max : float or None, optional
        If set, caps the colorbar maximum and clips all values above this to cbar_max.

    Returns:
    --------
    None
        The function displays the visualization but doesn't return any values.
    """
    n_rows = (n_components + n_columns - 1) // n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns + 1, 4 * n_rows))

    # Compute sqrt(MI) and optionally clip values above cbar_max
    all_components = np.array([np.sqrt(model.mis[i]).reshape(*component_shape) for i in range(n_components)])
    # all_components = np.array([model.mis[i].reshape(*component_shape) for i in range(n_components)])
    vmin = all_components.min()
    vmax = all_components.max() if cbar_max is None else cbar_max

    # Clip all values above vmax if cbar_max is set
    if cbar_max is not None:
        all_components = np.clip(all_components, vmin, cbar_max)

    for i in range(n_components):
        ax = axes[i // n_columns, i % n_columns]
        component = all_components[i]
        # im = ax.imshow(component, vmin=vmin, vmax=vmax)
        im = ax.imshow(component)
        ax.set_title(f'Component {i+1}')
        ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_columns):
        axes[j // n_columns, j % n_columns].axis('off')

    plt.tight_layout()
    fig.subplots_adjust(right=0.9)

    # # Add a colorbar that applies to all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Component Strength')

    plt.show()
