from typing import List

import torch.nn as nn


class MultilayerPerceptron(nn.Module):
    """Simple dense network in PyTorch.

    The number and shape of dense layers can be parametrized in the config file."""

    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int):
        super().__init__()
        layers = []
        previous_layer_size = input_size

        # Create dense layers 1 to N-1
        for layer_size in hidden_layers:
            linear = nn.Linear(
                in_features=previous_layer_size, out_features=layer_size, bias=True
            )
            nn.init.xavier_normal_(linear.weight, gain=1)
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.ReLU())
            previous_layer_size = layer_size

        # Create final layer
        linear = nn.Linear(
            in_features=previous_layer_size, out_features=output_size, bias=True
        )
        nn.init.xavier_normal_(linear.weight, gain=0.1)
        nn.init.constant_(linear.bias, 0.0)
        layers.append(linear)

        self.model = nn.Sequential(*layers)

    def forward(self, input_):
        return self.model(input_)
