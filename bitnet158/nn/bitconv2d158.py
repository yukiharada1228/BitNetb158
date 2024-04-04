from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.common_types import _size_2_t


class BitConv2d158(nn.Conv2d):
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
        num_bits: int = 8,
        device=None,
        dtype=None,
    ) -> None:
        super(BitConv2d158, self).__init__(
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
        self.num_bits = num_bits
        self.quantization_range = 2 ** (self.num_bits - 1)
        self.epsilon = 1e-6

    def absmax_quantize(
        self, x: torch.Tensor, quantization_range: int, epsilon: float
    ) -> Tuple[torch.Tensor, float]:
        gamma = torch.abs(x).max().clamp(min=epsilon)
        x_scaled = x * quantization_range / gamma
        x_q = (
            torch.clamp(
                torch.round(x_scaled), -quantization_range, quantization_range - 1
            )
            - x_scaled
        ).detach() + x_scaled
        return x_q, gamma

    def quantize_weights(
        self, weight: torch.Tensor, epsilon: float
    ) -> Tuple[torch.Tensor, float]:
        beta = weight.abs().mean().clamp(min=epsilon)
        weight_trinarized = (
            torch.clamp(torch.round(weight / beta), -1, 1) - weight
        ).detach() + weight
        return weight_trinarized, beta

    def dequantize(self, x: torch.Tensor, gamma: float, beta: float) -> torch.Tensor:
        return x * (beta * gamma / self.quantization_range)

    def forward(self, x):
        x_q, gamma = self.absmax_quantize(x, self.quantization_range, self.epsilon)
        w_q, beta = self.quantize_weights(self.weight, self.epsilon)
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
