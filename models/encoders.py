"""Shared CNN encoder building blocks (ResNet-style, stride-configurable).

A single CNNEncoder instance is used as the Siamese trunk (shared weights) in
Model 1, and one instance per modality in Model 2. Any torchvision/timm
backbone can be swapped in as long as it returns the same list of feature maps.
"""
from typing import List, Sequence

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Stem(nn.Module):
    """3x3 stride-2 conv + two stride-1 convs -> output stride 2."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(c_in, c_out, 3, 2),
            ConvBNAct(c_out, c_out, 3, 1),
            ConvBNAct(c_out, c_out, 3, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        idn = x if self.short is None else self.short(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + idn)


class Stage(nn.Module):
    def __init__(self, c_in: int, c_out: int, n_blocks: int, stride: int):
        super().__init__()
        blocks = [ResBlock(c_in, c_out, stride)]
        blocks += [ResBlock(c_out, c_out, 1) for _ in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class CNNEncoder(nn.Module):
    """ResNet-style multi-scale encoder.

    Args:
        in_channels: input channels (3 for RGB, >3 for multispectral, 1-2 for SAR).
        widths: channels of each returned stage.
        blocks: residual blocks per stage.
        out_strides: output stride of each returned stage relative to the input,
            e.g. (4, 8, 16, 32). Input H/W must be divisible by max(out_strides)
            (use (4, 8, 16) if you need more flexibility).

    Returns:
        List of feature maps, shallow -> deep. feats[i] has stride out_strides[i]
        and channels widths[i].
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: Sequence[int] = (64, 128, 256, 512),
        blocks: Sequence[int] = (2, 2, 2, 2),
        out_strides: Sequence[int] = (4, 8, 16, 32),
    ):
        super().__init__()
        assert len(widths) == len(blocks) == len(out_strides)
        stem_c = max(widths[0] // 2, 16)
        self.stem = Stem(in_channels, stem_c)  # stride 2
        self.out_strides = tuple(out_strides)
        self.out_channels = tuple(widths)
        stages, prev = [], stem_c
        for w, s, n in zip(widths, out_strides, blocks):
            stride = 1 if s <= 2 else 2  # stem already provides stride 2
            stages.append(Stage(prev, w, n, stride))
            prev = w
        self.stages = nn.ModuleList(stages)

    @property
    def max_stride(self) -> int:
        return max(self.out_strides)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats
