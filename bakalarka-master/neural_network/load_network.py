import os
import torch
from neural_network.training_network import network_definition


def load_model(path):
    """
    Loads a trained neural network model from a checkpoint file, reconstructs its architecture using saved
    hyperparameters, loads the trained weights, and returns the model in evaluation mode.

    Parameters:
        path (str): Path to the saved model checkpoint file.

    Returns:
        torch.nn.Module: The reconstructed and loaded PyTorch model, ready for inference.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist at the specified path.
        KeyError: If required keys are missing in the checkpoint.
        RuntimeError: If the model weights cannot be loaded into the reconstructed architecture.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at path: {path}")

    print("Model loaded from file.")

    # 1. Load the entire saved package (checkpoint) onto the CPU
    checkpoint = torch.load(path, map_location="cpu")

    # 2. Extract the hyperparameters used during training
    hp = checkpoint['hyperparameters']

    num_conv_layers = hp["num_conv_layers"]
    out_channels_list = [hp[f"out_channels_l{i}"] for i in range(num_conv_layers)]
    kernel_sizes = [hp[f"kernel_size_l{i}"] for i in range(num_conv_layers)]

    num_fc_layers = hp["num_fc_layers"]
    fc_neurons_list = [hp[f"fc_neurons_l{i}"] for i in range(num_fc_layers)]

    dropout_rate = hp["dropout_rate"]

    # 3. Initialize an empty network tailored exactly to these parameters
    model = network_definition.DynamicConvNN(
        num_conv_layers=num_conv_layers,
        out_channels_list=out_channels_list,
        kernel_sizes=kernel_sizes,
        fc_neurons_list=fc_neurons_list,
        dropout_rate=dropout_rate
    )

    # 4. Load the actual trained weights into the prepared network structure
    model.load_state_dict(checkpoint['state_dict'])

    # Switch to evaluation mode (disables Dropout, etc., essential for inference)
    model.eval()

    return model
