import torch.nn as nn

from .nn import BitLinearb158


def replace_linear_with_bitlinearb158(model):
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
        else:
            replace_linear_with_bitlinearb158(module)
