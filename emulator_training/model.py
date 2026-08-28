from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import torch
from torch import nn


def build_model(model_config: dict, input_dim: int, output_dim: int) -> nn.Module:
    model_type = model_config.get("type", "mlp")
    if model_type == "mlp":
        return build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=model_config["hidden_dims"],
            activation=model_config["activation"],
            dropout=float(model_config["dropout"]),
        )
    if model_type == "column_conv":
        return ColumnConvNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_channels=model_config["hidden_channels"],
            kernel_size=int(model_config.get("kernel_size", 5)),
            activation=model_config["activation"],
            dropout=float(model_config["dropout"]),
        )
    raise KeyError(f"Unsupported model type {model_type!r}.")


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Iterable[int],
    activation: str = "gelu",
    dropout: float = 0.0,
) -> nn.Module:
    activation_layer = get_activation(activation)
    layers: list[nn.Module] = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(activation_layer())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = int(hidden_dim)

    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class ColumnConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_channels: Sequence[int],
        kernel_size: int = 5,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ColumnConvNet requires an odd kernel size for symmetric padding.")

        activation_layer = get_activation(activation)
        layers: list[nn.Module] = []
        in_channels = input_dim
        padding = kernel_size // 2

        for hidden_channel in hidden_channels:
            layers.append(nn.Conv1d(in_channels, int(hidden_channel), kernel_size=kernel_size, padding=padding))
            layers.append(activation_layer())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_channels = int(hidden_channel)

        layers.append(nn.Conv1d(in_channels, output_dim, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"ColumnConvNet expects input with shape (batch, z, feature), got {tuple(inputs.shape)}")
        column_major = inputs.transpose(1, 2)
        outputs = self.network(column_major)
        return outputs.transpose(1, 2)


def get_activation(name: str) -> type[nn.Module]:
    registry = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if name not in registry:
        raise KeyError(f"Unsupported activation {name!r}.")
    return registry[name]
