import torch.nn as nn

from .nn import BitLinearb158


def convert_weights_to_packed(model, target_layer: nn.Module):
    assert target_layer in (BitLinearb158,), "Layer must be BitLinearb158"
    model.eval()
    for module in model.modules():
        if isinstance(module, target_layer):
            module.convert_weights_to_packed()
