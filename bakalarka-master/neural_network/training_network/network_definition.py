import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from neural_network.training_network import TRAINING_CONSTANTS
except ImportError:
    import TRAINING_CONSTANTS


class DynamicConvNN(nn.Module):
    """
     Defines a configurable convolutional neural network for image classification with a dynamic architecture.

     The network consists of a variable number of convolutional and pooling layers, followed by fully connected layers,
     and an output layer for binary classification (2 classes). The architecture is determined by the input parameters.

     Parameters:
         num_conv_layers (int): Number of convolutional layers.
         out_channels_list (list[int]): List of output channels for each convolutional layer.
         kernel_sizes (list[int]): List of kernel sizes for each convolutional layer.
         fc_neurons_list (list[int]): List of neuron counts for each fully connected (FC) layer.
         dropout_rate (float, optional): Dropout probability for FC layers (default: 0.2).

     Methods:
         forward(x):
             Performs a forward pass through the network.

             Parameters:
                 x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).

             Returns:
                 torch.Tensor: Log-probabilities for each class (shape: [batch_size, 2]).
     """

    def __init__(self, num_conv_layers, out_channels_list, kernel_sizes, fc_neurons_list, dropout_rate=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_channels = 1

        for i in range(num_conv_layers):
            k_size = kernel_sizes[i]
            out_c = out_channels_list[i]
            self.convs.append(nn.Conv2d(in_channels, out_c, kernel_size=k_size, stride=1, padding=k_size // 2))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_c

        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, TRAINING_CONSTANTS.CELL_WIDTH, TRAINING_CONSTANTS.CELL_HEIGHT)
            for conv, pool in zip(self.convs, self.pools):
                dummy_x = pool(F.relu(conv(dummy_x)))
            flattened_size = dummy_x.flatten(start_dim=1).shape[1]

        self.fcs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = flattened_size

        for hidden_neurons in fc_neurons_list:
            self.fcs.append(nn.Linear(in_features, hidden_neurons))
            self.dropouts.append(nn.Dropout(p=dropout_rate))
            in_features = hidden_neurons

        self.out_layer = nn.Linear(in_features, 2)

    def forward(self, x):
        for conv, pool in zip(self.convs, self.pools):
            x = pool(F.relu(conv(x)))

        x = x.flatten(start_dim=1)

        for fc, drop in zip(self.fcs, self.dropouts):
            x = F.relu(fc(x))
            x = drop(x)

        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)
