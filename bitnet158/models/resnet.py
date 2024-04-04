from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from bitnet158.nn import BitConv2d158, BitLinear158

__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


def conv3x3(
    in_planes: int,
    conv_layer: nn.Module,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Module:
    """3x3 convolution with padding"""
    return conv_layer(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(
    in_planes: int, conv_layer: nn.Module, out_planes: int, stride: int = 1
) -> nn.Module:
    """1x1 convolution"""
    return conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        conv_layer: nn.Module,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, conv_layer, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, conv_layer, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        conv_layer: nn.Module,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, conv_layer, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, conv_layer, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, conv_layer, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        linear_layer: Callable,
        conv_layer: Callable,
        block: BasicBlock | Bottleneck,
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.linear_layer = linear_layer
        self.conv_layer = conv_layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv_layer(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], conv_layer)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            conv_layer,
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            conv_layer,
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            conv_layer,
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = linear_layer(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, conv_layer):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: BasicBlock | Bottleneck,
        planes: int,
        blocks: int,
        conv_layer: nn.Module,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, conv_layer, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                conv_layer,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    conv_layer,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    @property
    def __name__(self) -> str:
        if self.linear_layer == BitLinear158 or self.conv_layer == BitConv2d158:
            return "BitNet158"
        return "FloatNet"

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    linear_layer: Callable,
    conv_layer: Callable,
    block: BasicBlock | Bottleneck,
    layers: list[int],
    **kwargs,
) -> ResNet:

    model = ResNet(linear_layer, conv_layer, block, layers, **kwargs)

    return model


def load_pretrained_weights(
    model, model_name: str, linear_layer, conv_layer, pretrained: bool, **kwargs
):
    if not pretrained:
        return model

    pretrained_model = torch.hub.load(
        "pytorch/vision:v0.10.0", model_name, pretrained=True
    )

    if pretrained_model.fc.out_features != model.fc.out_features:
        state_dict = pretrained_model.state_dict()
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

    return model


def resnet18(linear_layer, conv_layer, pretrained: bool, **kwargs):
    model = _resnet(linear_layer, conv_layer, BasicBlock, [2, 2, 2, 2], **kwargs)
    return load_pretrained_weights(
        model, "resnet18", linear_layer, conv_layer, pretrained, **kwargs
    )


def resnet34(linear_layer, conv_layer, pretrained: bool, **kwargs):
    model = _resnet(linear_layer, conv_layer, BasicBlock, [3, 4, 6, 3], **kwargs)
    return load_pretrained_weights(
        model, "resnet34", linear_layer, conv_layer, pretrained, **kwargs
    )


def resnet50(linear_layer, conv_layer, pretrained: bool, **kwargs):
    model = _resnet(linear_layer, conv_layer, Bottleneck, [3, 4, 6, 3], **kwargs)
    return load_pretrained_weights(
        model, "resnet50", linear_layer, conv_layer, pretrained, **kwargs
    )


def resnet101(linear_layer, conv_layer, pretrained: bool, **kwargs):
    model = _resnet(linear_layer, conv_layer, Bottleneck, [3, 4, 23, 3], **kwargs)
    return load_pretrained_weights(
        model, "resnet101", linear_layer, conv_layer, pretrained, **kwargs
    )


def resnet152(linear_layer, conv_layer, pretrained: bool, **kwargs):
    model = _resnet(linear_layer, conv_layer, Bottleneck, [3, 8, 36, 3], **kwargs)
    return load_pretrained_weights(
        model, "resnet152", linear_layer, conv_layer, pretrained, **kwargs
    )


def main():
    _ = resnet18(BitLinear158, BitConv2d158, pretrained=True, num_classes=100)
    _ = resnet34(BitLinear158, BitConv2d158, pretrained=True, num_classes=100)
    _ = resnet50(BitLinear158, BitConv2d158, pretrained=True, num_classes=100)
    _ = resnet101(BitLinear158, BitConv2d158, pretrained=True, num_classes=100)
    _ = resnet152(BitLinear158, BitConv2d158, pretrained=True, num_classes=100)


if __name__ == "__main__":
    main()
