from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.common_types import _size_2_t


class QuantizationMixin:
    def __init__(self, num_bits: int = 8, epsilon: float = 1e-5):
        self.num_bits = num_bits
        self.quantization_range = 2 ** (self.num_bits - 1)
        self.epsilon = epsilon

    def absmax_quantize(
        self, x: torch.Tensor, quantization_range: int, epsilon: float
    ) -> Tuple[torch.Tensor, float]:
        gamma = x.abs().max().clamp(min=epsilon)
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
        weight_scaled = weight / beta
        weight_quantized = (
            torch.clamp(torch.round(weight_scaled), -1, 1) - weight_scaled
        ).detach() + weight_scaled
        return weight_quantized, beta

    def dequantize(self, x: torch.Tensor, gamma: float, beta: float) -> torch.Tensor:
        return x * (beta * gamma / self.quantization_range)

    def pack_ternary(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] % 4 == 0
        ), "The last dimension size of x must be divisible by 4"
        dtype = torch.int8
        device = x.device
        x_mapped = x.to(dtype).clone()
        x_mapped[x == -1] = 2
        shift = torch.arange(4, device=x.device) * 2
        shape = x.shape[:-1]
        x = x_mapped.view(-1, x.shape[-2], x.shape[-1] // 4, 4)
        x = x << shift[None, None, None, :]
        x = x.sum(-1)
        x = x.view(*shape, *x.shape[-1:])
        return x.to(dtype).to(device)

    def unpack_ternary(self, x: torch.Tensor) -> torch.Tensor:
        masks = (3 << (2 * torch.arange(4, device=x.device))).view(1, 1, 1, -1)
        x_expanded = x.unsqueeze(-1)
        x_expanded = x_expanded * torch.ones_like(masks)
        unpacked = (x_expanded & masks) >> (2 * torch.arange(4, device=x.device)).view(
            1, 1, 1, -1
        )
        unpacked = torch.where(
            unpacked == 2, torch.tensor(-1, device=x.device), unpacked
        )
        return unpacked.view(*x.shape[:-1], -1)

    def convert_weights_to_packed(self):
        if isinstance(self.weight, torch.nn.Parameter):
            w_q, beta = self.quantize_weights(self.weight, self.epsilon)
            packed_weight = self.pack_ternary(w_q)
            self.beta = torch.nn.Parameter(beta)
            del self.weight
            self.register_buffer("weight", packed_weight)


class BitLinearb158(nn.Linear, QuantizationMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 8,
        epsilon: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super(BitLinearb158, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )
        QuantizationMixin.__init__(self, num_bits, epsilon)

    def forward(self, x):
        x_norm = F.layer_norm(x, x.shape[1:])
        x_q, gamma = self.absmax_quantize(x_norm, self.quantization_range, self.epsilon)
        if isinstance(self.weight, torch.nn.Parameter):
            w_q, beta = self.quantize_weights(self.weight, self.epsilon)
        else:
            w_q, beta = self.unpack_ternary(self.weight).to(torch.float32), self.beta
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
        num_bits: int = 8,
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
        QuantizationMixin.__init__(self, num_bits, epsilon)

    def forward(self, x):
        x_norm = F.layer_norm(x, x.shape[1:])
        x_q, gamma = self.absmax_quantize(x_norm, self.quantization_range, self.epsilon)
        if isinstance(self.weight, torch.nn.Parameter):
            w_q, beta = self.quantize_weights(self.weight, self.epsilon)
        else:
            w_q, beta = self.unpack_ternary(self.weight).to(torch.float32), self.beta
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
