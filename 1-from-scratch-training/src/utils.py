import torch
import torch.nn as nn


def get_output_shape(input_size: tuple, layer: nn.Module) -> tuple:
    """
    Restituisce la shape del tensore in uscita da un layer.
    """
    with torch.no_grad():
        x = torch.zeros(input_size)
        return tuple(layer(x).shape)
