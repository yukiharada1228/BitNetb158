import torch.nn as nn
import torchvision

from ..replace_hf import replace_layers_with_bitb158_layers


def bitresnet18b158(num_classes=10):
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    replace_layers_with_bitb158_layers(model)
    return model
