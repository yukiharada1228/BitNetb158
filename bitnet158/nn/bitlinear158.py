from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class BitLinear158(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 8,
        is_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super(BitLinear158, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )
        self.num_bits = num_bits
        self.quantization_range = 2 ** (self.num_bits - 1)
        self.epsilon = 1e-6
        if is_norm:
            self.layer_norm = LlamaRMSNorm(hidden_size=in_features, eps=rms_norm_eps)
        else:
            self.layer_norm = nn.Identity()

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
        x_norm = self.layer_norm(x)
        x_q, gamma = self.absmax_quantize(x_norm, self.quantization_range, self.epsilon)
        w_q, beta = self.quantize_weights(self.weight, self.epsilon)
        x_matmul = F.linear(x_q, w_q, self.bias)
        output = self.dequantize(x_matmul, gamma, beta)
        return output
