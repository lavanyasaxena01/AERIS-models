"""U-Net style decoder with skip connections, back to full input resolution."""
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetDecoder(nn.Module):
    """Consumes multi-scale features ordered SHALLOW -> DEEP and produces
    full-resolution change logits [B, 1, H, W].

    Skip connections come from every encoder scale, which is what preserves
    fine change boundaries. The shallowest feature stride (encoder_strides[0])
    determines the final upsampling factor, so the output always matches the
    input HxW exactly.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (64, 128, 256, 512),
        encoder_strides: Sequence[int] = (4, 8, 16, 32),
        decoder_channels: int = 256,
    ):
        super().__init__()
        blocks = []
        c_in = encoder_channels[-1]
        for i in range(len(encoder_channels) - 2, -1, -1):
            blocks.append(DecoderBlock(c_in, encoder_channels[i], decoder_channels))
            c_in = decoder_channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(decoder_channels, 1, 1)
        self.out_stride = int(encoder_strides[0])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for block, skip in zip(self.blocks, reversed(feats[:-1])):
            x = block(x, skip)
        logits = self.head(x)
        return F.interpolate(logits, scale_factor=self.out_stride, mode="bilinear", align_corners=False)
