from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..triton_kernels.bitmat_kernel import batched_bitmat


class QuantizationMixin:
    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon

    def absmax_quantize(
        self,
        x: torch.Tensor,
        ste: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        gamma = x.abs().max().clamp(min=self.epsilon)
        x_scaled = x * 127 / gamma
        if ste:
            x_q = (
                torch.clamp(x_scaled, -128, 127).to(torch.int8).to(x_scaled.dtype)
                - x_scaled
            ).detach() + x_scaled
        else:
            x_q = torch.clamp(x_scaled, -128, 127).to(torch.int8)
        return x_q, gamma

    def quantize_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
        beta = weight.abs().mean().clamp(min=self.epsilon)
        weight_scaled = weight / beta
        weight_quantized = (
            torch.clamp(weight_scaled, -1, 1).to(torch.int8).to(weight.dtype)
            - weight
        ).detach() + weight
        return weight_quantized, beta

    def dequantize(self, x: torch.Tensor, gamma: float, beta: float) -> torch.Tensor:
        return x * (beta * gamma / 127)

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
        return unpacked.view(*x.shape[:-1], -1).to(torch.int8)

    def convert_weights_to_packed(self):
        if isinstance(self.weight, torch.nn.Parameter):
            w_q, beta = self.quantize_weights(self.weight)
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
            x_q, gamma = self.absmax_quantize(x_norm)
            w_q, beta = self.quantize_weights(self.weight)
            x_matmul = F.linear(x_q, w_q, self.bias)
        else:
            x_q, gamma = self.absmax_quantize(x_norm, ste=False)
            if x_q.dim() == 2:
                x_q = x_q.unsqueeze(1)
            x_matmul = batched_bitmat(x_q, self.weight)
            if self.bias is not None:
                x_matmul += self.bias.unsqueeze(0).expand_as(x_matmul)
            beta = self.beta
            # w_q, beta = self.unpack_ternary(self.weight), self.beta
            # x_matmul = F.linear(
            #     x_q.to(torch.float32), w_q.to(torch.float32), self.bias
            # )
        output = self.dequantize(x_matmul, gamma, beta)
        return output
