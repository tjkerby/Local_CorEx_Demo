import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

from mnist_classifier.model import MLPClassifier, Autoencoder
from pytorch_lightning import LightningModule


def reconstruct(model, input):
    '''
    Reconstructs the input data using the encoder-decoder architecture of the given model.
    
    This function passes the input through the model's encoder to obtain the latent representation, 
    then reconstructs the input by passing the latent representation through the decoder.
    
    Args:
    - model: the trained autoencoder model (expected to have `encoder` and `decoder` attributes).
    - input: the data to be reconstructed. Should be a tensor of any shape.
    
    Returns:
    - reconstructed_input: a numpy array containing the reconstructed version of the input data.
    
    Behavior:
    - If the input is a batch (i.e., multi-dimensional), it reshapes the input to a 2D tensor 
      with shape `(batch_size, -1)` before passing it through the model.
    - The reconstruction is detached from the computation graph and returned as a NumPy array.
    '''
    
    if len(input.shape) > 1:
        bs = input.shape[0]
    else:
        bs = 1
    return model.decoder(model.encoder(input.reshape(bs,-1))).detach().numpy()

def prune_model_node(model, hidden_layer_idx, node_indexes):
    """
    Prunes specific nodes (neurons) in the specified hidden layer of the model by zeroing out their weights and biases.
    
    Args:
        model: The neural network model (assumed to have a `hidden_layers` attribute, an nn.ModuleList).
        hidden_layer_idx: Integer index of the hidden layer to prune.
        node_indexes: Iterable of node indices to zero out.
        
    Returns:
        model: The modified model with pruned nodes.
    """
    # For a dynamic model, the layer is stored in model.hidden_layers (an nn.ModuleList)
    if hasattr(model, 'hidden_layers'):
        layer = model.hidden_layers[hidden_layer_idx]
        # Zero out the corresponding rows in the weight matrix and biases
        for node in node_indexes:
            # Ensure the weight tensor is two-dimensional.
            if layer.weight.data.dim() >= 2:
                layer.weight.data[node, :] = 0
            if layer.bias is not None:
                layer.bias.data[node] = 0
    return model

def get_hidden_states(model, data_module, num_layers, device, input_size=784):
    '''
    Extracts and returns the hidden layer representations of input data for each layer in a pretrained neural network.

    Args:
    - model: The trained model from which to extract hidden layer outputs.
    - data_module: Module providing the data loader for prediction.
    - num_layers: The number of hidden layers in the model.
    - device: The device ('cpu' or 'cuda') on which computations will be performed.
    - input_size: The size of the input data (default: 784, for flattened 28x28 images).

    Returns:
    - c_predictions: Concatenated predictions from the model's output layer.
    - *hidden_states: Concatenated hidden layer representations for each hidden layer (one per layer).
    - c_inputs: Concatenated input data for all batches, reshaped as specified by input_size.
    - c_labels: Concatenated true labels for the inputs, collected from all batches.
    '''
    model.eval()
    predictions = []
    hidden_states = [[] for _ in range(num_layers)]
    all_inputs = []
    all_labels = []

    predict_loader = data_module.predict_dataloader()
    for batch in predict_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            result = model.predict_with_hidden(inputs)
            output = result[0]
            hidden_layers = result[1]

        predictions.append(output.cpu().numpy())
        for i, hidden_layer in enumerate(hidden_layers):
            hidden_states[i].append(hidden_layer.view(hidden_layer.size(0), -1).cpu().numpy())

        all_inputs.append(inputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    c_predictions = np.concatenate(predictions, axis=0)
    c_hidden_states = [np.concatenate(hidden_state, axis=0) for hidden_state in hidden_states]
    c_inputs = np.concatenate(all_inputs, axis=0)
    c_labels = np.concatenate(all_labels, axis=0)

    return c_predictions, *c_hidden_states, c_inputs.reshape(-1, input_size), c_labels

def build_encoder_from_classifier(classifier):
    """
    Build an encoder as an nn.Sequential by chaining all the hidden layers (and their corresponding
    batch norm, activation, and dropout) from the classifier.
    """
    encoder_modules = []
    for i, linear_layer in enumerate(classifier.hidden_layers):
        encoder_modules.append(linear_layer)
        if classifier.batch_norms is not None:
            encoder_modules.append(classifier.batch_norms[i])
        # For the first layer, optionally skip activation if specified in config
        if i == 0 and classifier.conf.get('no_act_1st_layer', False):
            pass
        else:
            encoder_modules.append(nn.ReLU())
        # Use the same dropout as in the classifier
        encoder_modules.append(classifier.dropout)
    return nn.Sequential(*encoder_modules)

def build_decoder_from_classifier(classifier):
    """
    Infer the decoder architecture as the mirror of the classifier's encoder.
    It uses the input dimension (assumed 28*28 for MNIST) and the output features
    of each encoder linear layer to form a list of dimensions, then builds a series of
    linear layers in reverse order. For each hidden decoder layer (i.e. not the final
    reconstruction layer), we optionally add batch norm, ReLU, and dropout.
    """
    # Get dropout and batch norm setting from classifier config
    dropout_p = classifier.conf.get('drop_out_p', 0.0)
    use_bn = classifier.conf.get('use_batch_norm', False)
    
    # Compute the dimensions used in the encoder.
    # The encoder starts at input_dim (28*28) and goes through each hidden layer.
    dims = [28 * 28]
    for linear_layer in classifier.hidden_layers:
        dims.append(linear_layer.out_features)
    # dims now is: [input_dim, h1, h2, ..., h_n]
    
    decoder_modules = []
    # Build decoder by iterating over dims in reverse order.
    # For each pair dims[i] -> dims[i-1], add a Linear layer.
    # For intermediate layers (not the final reconstruction), add bn, ReLU and dropout.
    for i in range(len(dims) - 1, 0, -1):
        in_features = dims[i]
        out_features = dims[i - 1]
        decoder_modules.append(nn.Linear(in_features, out_features))
        # If not the final layer, add extra modules.
        if i - 1 > 0:
            if use_bn:
                decoder_modules.append(nn.BatchNorm1d(out_features))
            decoder_modules.append(nn.ReLU())
            if dropout_p > 0.0:
                decoder_modules.append(nn.Dropout(dropout_p))
    # Final activation: Sigmoid to squash outputs to [0,1] (suitable for MNIST)
    decoder_modules.append(nn.Sigmoid())
    
    return nn.Sequential(*decoder_modules)

def load_models(ae_ckpt: str, classifier_ckpt: str, ae_conf: dict, clf_conf: dict) -> LightningModule:
    """
    Load an autoencoder model whose encoder is initialized from a pre-trained classifier.

    Parameters:
    - base_path (str): The directory containing the model checkpoints and configuration files.
    - ae_conf (dict): Configuration dictionary for the autoencoder.
    - clf_conf (dict): Configuration dictionary for the classifier.

    Returns:
    - LightningModule: The loaded autoencoder model.
    """

    classifier = MLPClassifier.load_from_checkpoint(classifier_ckpt, conf=clf_conf)
    encoder = build_encoder_from_classifier(classifier)
    decoder = build_decoder_from_classifier(classifier)
    autoencoder = Autoencoder(ae_conf)
    autoencoder.encoder = encoder
    autoencoder.decoder = decoder
    state_dict = torch.load(ae_ckpt)['state_dict']
    autoencoder.load_state_dict(state_dict)

    return autoencoder, classifier

def compute_cluster_accuracies(clf, loader, device, indexes, verbose=False, return_probs=True):
    preds = []
    labels = []
    
    clf.eval()
    with torch.no_grad():
        for batch_idx, (image_batch, label_batch) in enumerate(tqdm(loader)):
            images = image_batch.to(device).float()
            pred_batch = clf(images)
            preds.append(pred_batch.cpu())
            labels.append(label_batch)
    
    preds_tensor = torch.cat(preds, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    probs = F.softmax(preds_tensor, dim=1)
    pred_label = probs.argmax(dim=1)
    
    base_accuracies = []
    for i in range(len(indexes)):
        base_accuracies.append((100*torch.sum(pred_label[indexes[i]] == labels_tensor[indexes[i]])/ pred_label[indexes[i]].size(0)).item())
        if verbose:
            print(i, np.round((100*torch.sum(pred_label[indexes[i]] == labels_tensor[indexes[i]])/ pred_label[indexes[i]].size(0)).item(), 2))
    if return_probs:
        return (base_accuracies, probs)
    return base_accuracies

def prune_resnet_fc_layer(model, node_indexes):
    """
    Prunes nodes in ResNet18's final fully connected layer by zeroing weights/biases.
    
    Args:
        model: ResNet18 model instance
        node_indexes: Iterable of node indices to prune (0-999 for ImageNet)
        
    Returns:
        Modified ResNet18 model with specified nodes disabled
    """
    # Access the final fully connected layer
    fc_layer = model.fc
    
    # Zero out specified nodes
    with torch.no_grad():  # Ensure no gradient tracking
        for node in node_indexes:
            # fc_layer.weight.data[node] = 0  # This is actually a row operation
            if fc_layer.weight.data.dim() >= 2:
                fc_layer.weight.data[:, node] = 0
            # if fc_layer.bias is not None:
            #     fc_layer.bias.data[node] = 0
                
    return model
