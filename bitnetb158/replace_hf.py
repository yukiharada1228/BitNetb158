import torch.nn as nn

from .nn import BitConv2db158, BitLinearb158


def replace_layers_with_bitb158_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            weight_device = module.weight.device
            weight_dtype = module.weight.dtype

            setattr(
                model,
                name,
                BitLinearb158(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                    device=weight_device,
                    dtype=weight_dtype,
                ),
            )
        elif isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias is not None
            padding_mode = module.padding_mode
            weight_device = module.weight.device
            weight_dtype = module.weight.dtype

            setattr(
                model,
                name,
                BitConv2db158(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode,
                    device=weight_device,
                    dtype=weight_dtype,
                ),
            )
        else:
            replace_layers_with_bitb158_layers(module)
