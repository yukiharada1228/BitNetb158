from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.common_types import _size_2_t


class QuantizationMixin:
    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon

    def absmax_quantize(
        self, x: torch.Tensor, epsilon: float
    ) -> Tuple[torch.Tensor, float]:
        gamma = x.abs().max().clamp(min=epsilon)
        x_scaled = x * 127 / gamma
        x_q = (torch.clamp(x_scaled, -128, 127) - x_scaled).detach() + x_scaled
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
        epsilon: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super(BitLinearb158, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )
        QuantizationMixin.__init__(self, epsilon)

    def forward(self, x):
        x_norm = F.layer_norm(x, x.shape[1:])
        if isinstance(self.weight, torch.nn.Parameter):
            x_q, gamma = self.absmax_quantize(x_norm, self.epsilon)
            w_q, beta = self.quantize_weights(self.weight, self.epsilon)
            x_matmul = F.linear(x_q, w_q, self.bias)
        else:
            gamma = x.abs().max().clamp(min=self.epsilon)
            x_scaled = x * 127 / gamma
            x_q = (torch.clamp(x_scaled, -127, 128)).to(torch.int8)
            w_q, beta = self.unpack_ternary(self.weight), self.beta
            x_matmul = F.linear(x_q.to(torch.float32), w_q.to(torch.float32), self.bias)
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
        x_norm = F.layer_norm(x, x.shape[1:])
        x_q, gamma = self.absmax_quantize(x_norm, self.epsilon)
        if isinstance(self.weight, torch.nn.Parameter):
            x_q, gamma = self.absmax_quantize(x_norm, self.epsilon)
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
        else:
            gamma = x.abs().max().clamp(min=self.epsilon)
            x_scaled = x * 127 / gamma
            x_q = (torch.clamp(x_scaled, -127, 128)).to(torch.int8)
            w_q, beta = self.unpack_ternary(self.weight), self.beta
            x_conv2d = F.conv2d(
                input=x_q.to(torch.float32),
                weight=w_q.to(torch.float32),
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        output = self.dequantize(x_conv2d, gamma, beta)
        return output
