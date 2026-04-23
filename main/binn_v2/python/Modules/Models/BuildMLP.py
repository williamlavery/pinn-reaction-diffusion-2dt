"""Build a configurable multilayer perceptron used by BINN components.

`BuildMLP` is the active MLP constructor in this repository. It supports a
bias-only mode (`use_single_bias`) used when learning constant coefficients.
"""


import torch
import torch.nn as nn


class BuildMLP(nn.Module):
    """
    Builds a standard multilayer perceptron (MLP) with configurable options for activation functions,
    dropout, and output layer characteristics. It also supports a bias-only mode.

    This class supports custom initialization and flexible activation functions for hidden and 
    output layers. It allows setting a random seed for reproducibility. Note: if `use_single_bias` 
    is set to True, the MLP construction is bypassed entirely in favor of a single learnable scalar.

    Attributes:
        use_single_bias (bool): If True, model skips MLP construction and acts as a single learnable bias.
        bias (nn.Parameter): The learnable scalar parameter, created only if `use_single_bias=True`.
        input_features (int): Number of input features for the MLP.
        layers (list[int]): List of integers specifying the size of each layer.
        activation (nn.Module): Activation function for hidden layers. Defaults to nn.Sigmoid().
        linear_output (bool): If True, the output layer has no activation. Defaults to True.
        output_activation (nn.Module): Activation function for the output layer.
        dropout_rate (float): Dropout rate applied to hidden layers. Defaults to 0.0 (no dropout).
        seed (int): Random seed for reproducibility. Defaults to 0.
        MLP (nn.Sequential): The constructed sequential network (only if `use_single_bias=False`).

    Args:
        input_features (int): Number of input features.
        layers (list[int]): List of integers specifying the size of each layer.
        activation (nn.Module, optional): Activation function for hidden layers. Defaults to nn.Sigmoid().
        linear_output (bool, optional): If True, output layer is linear. Defaults to True.
        output_activation (nn.Module, optional): Activation function for the output layer. Defaults to None.
        dropout_rate (float, optional): Dropout rate for hidden layers. Defaults to 0.0.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        use_single_bias (bool, optional): If True, bypasses MLP to return a single learnable scalar. Defaults to False.

    Methods:
        forward(x):
            Computes the forward pass of the MLP (or broadcasts the single bias).

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, input_features).

    Returns:
        torch.Tensor: Output tensor. If using single bias, shape is (batch_size, 1). 
                      Otherwise, shape is (batch_size, layers[-1]).
    """
    
    def __init__(self, 
                 input_features, 
                 layers, 
                 activation=None, 
                 linear_output=True,
                 output_activation=None,
                 dropout_rate=0.0,
                 seed=0,
                 use_single_bias=False):
        """
        Initializes the BuildMLP class with the specified configurations.

        Args:
            input_features (int): Number of input features.
            layers (list[int]): List of integers specifying the size of each layer.
            activation (nn.Module, optional): Activation function for hidden layers.
            linear_output (bool, optional): If True, output layer is linear.
            output_activation (nn.Module, optional): Activation function for the output layer.
            dropout_rate (float, optional): Dropout rate for hidden layers.
            seed (int, optional): Random seed for reproducibility.
            use_single_bias (bool, optional): If True, model has only one learnable bias parameter.
        """
        super().__init__()

        self.use_single_bias = use_single_bias

        if use_single_bias:
            # Single learnable scalar bias (broadcasted in forward)
            self.bias = nn.Parameter(torch.zeros(1))
            return  # Skip the rest of the MLP construction

        self.input_features = input_features
        self.layers = layers
        self.seed = seed
        self.activation = activation if activation is not None else nn.Sigmoid()
        self.linear_output = linear_output
        self.output_activation = output_activation if output_activation else self.activation
        self.dropout_rate = dropout_rate

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        operations = []
        for i, layer in enumerate(layers[:-1]):
            operations.append(nn.Linear(in_features=self.input_features, out_features=layer, bias=True))
            self.input_features = layer
            operations.append(self.activation)
            if self.dropout_rate > 0:
                operations.append(nn.Dropout(p=self.dropout_rate))

        # Output layer
        operations.append(nn.Linear(in_features=self.input_features, out_features=layers[-1], bias=True))
        if not self.linear_output:
            operations.append(self.output_activation)

        self.MLP = nn.Sequential(*operations)

    def forward(self, x):
        """
        Forward pass of the MLP or bias-only model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features).

        Returns:
            torch.Tensor: Output tensor. If use_single_bias is True, returns shape (batch_size, 1). 
                          Otherwise returns shape (batch_size, layers[-1]).
        """
        if self.use_single_bias:
            return self.bias.expand(x.size(0), 1)  # broadcast bias to batch size
        return self.MLP(x)