from typing import Union

import torch.nn.functional as F
from torch import nn
from torch.nn.common_types import _size_2_t


def activation_quant(x):
    """Per-token quantization to 8 bits. No grouping is needed for quantization.
    Args:
    x: an activation tensor with shape [n, d]
    Returns:
    y: a quantized activation tensor with shape [n, d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    """Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
    w: a weight tensor with shape [d, k]
    Returns:
    u: a quantized weight with shape [d, k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


class BitLinearb158(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super(BitLinearb158, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )
        self.layer_norm = nn.LayerNorm(in_features, eps=1e-5)

    def forward(self, x):
        """
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        w = self.weight  # a weight tensor with shape [d, k]
        x_norm = self.layer_norm(x)
        # A trick for implementing Straight−Through−Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y


class BitConv2db158(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        epsilon: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super(BitConv2db158, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        w = self.weight
        x_norm = F.layer_norm(x, x.shape[1:], eps=1e-5)
        # A trick for implementing Straight−Through−Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.conv2d(
            input=x_quant,
            weight=w_quant,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return y
