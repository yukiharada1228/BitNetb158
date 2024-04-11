from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.common_types import _size_2_t


class QuantizationMixin:
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def absmax_quantize(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        gamma = x.abs().max().clamp(min=self.epsilon)
        x_scaled = x * 127 / gamma
        x_q = (x_scaled.round().clamp(-128, 127) - x_scaled).detach() + x_scaled
        return x_q, gamma

    def quantize_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
        beta = weight.abs().mean().clamp(min=self.epsilon)
        weight_scaled = weight / beta
        weight_quantized = (
            weight_scaled.round().clamp(-1, 1) - weight
        ).detach() + weight
        return weight_quantized, beta

    def dequantize(self, x: torch.Tensor, gamma: float, beta: float) -> torch.Tensor:
        return x * (beta * gamma / 127)


class BitLinearb158(nn.Linear, QuantizationMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        epsilon: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super(BitLinearb158, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )
        QuantizationMixin.__init__(self, epsilon)
        self.layernorm = nn.LayerNorm(in_features, eps=epsilon)

    def forward(self, x):
        x_norm = self.layernorm(x)
        x_q, gamma = self.absmax_quantize(x_norm)
        w_q, beta = self.quantize_weights(self.weight)
        x_matmul = F.linear(x_q, w_q, self.bias)
        output = self.dequantize(x_matmul, gamma, beta)
        return output


class BitConv2db158(nn.Conv2d, QuantizationMixin):
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
        QuantizationMixin.__init__(self, epsilon)

    def forward(self, x):
        x_norm = F.layer_norm(x, x.shape[1:], eps=self.epsilon)
        x_q, gamma = self.absmax_quantize(x_norm)
        w_q, beta = self.quantize_weights(self.weight)
        x_conv2d = F.conv2d(
            input=x_q,
            weight=w_q,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        output = self.dequantize(x_conv2d, gamma, beta)
        return output
