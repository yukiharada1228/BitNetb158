import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BitLinear158(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 8,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.eps: float = 1e-5
        self.quantization_range: int = 2 ** (num_bits - 1)

    def ste_weights(self, weights_gamma: float) -> Tensor:
        scaled_weights: Tensor = self.weight / (weights_gamma + self.eps)
        bin_weights_no_grad: Tensor = torch.sign(scaled_weights) * torch.clamp(
            torch.abs(scaled_weights).round(), max=1.0
        )
        bin_weights_with_grad: Tensor = (
            bin_weights_no_grad - self.weight
        ).detach() + self.weight
        return bin_weights_with_grad

    def binarize_weights(self, weights_gamma: float) -> Tensor:
        binarized_weights = self.ste_weights(weights_gamma)
        return binarized_weights

    def quantize_activations(self, input: Tensor, input_gamma: float) -> Tensor:
        quantized_input = torch.clamp(
            input * self.quantization_range / input_gamma,
            -self.quantization_range + self.eps,
            self.quantization_range - self.eps,
        )
        return quantized_input

    def dequantize_activations(
        self, input: Tensor, input_gamma: float, beta: float
    ) -> Tensor:
        return input * input_gamma * beta / self.quantization_range

    def forward(self, input: Tensor) -> Tensor:
        normalized_input: Tensor = F.layer_norm(input, (input.shape[1:]))
        input_gamma: float = normalized_input.abs().max().item()
        weight_abs_mean: float = self.weight.abs().mean().item()

        binarized_weights = self.binarize_weights(weight_abs_mean)
        input_quant = self.quantize_activations(normalized_input, input_gamma)
        output = F.linear(input_quant, binarized_weights, self.bias)
        output = self.dequantize_activations(output, input_gamma, weight_abs_mean)

        return output
