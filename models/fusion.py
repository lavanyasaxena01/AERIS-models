"""Temporal interaction and multimodal fusion modules."""
import torch
import torch.nn as nn


def _cbr(c_in: int, c_out: int, k: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class DifferenceFusion(nn.Module):
    """Rich temporal difference representation.

    Builds [F1, F2, F1-F2, |F1-F2|] (4C channels), fuses with 1x1 + 3x3 convs,
    and adds the result as a learned residual on top of the plain difference.
    The network therefore keeps the well-behaved linear difference while being
    free to learn non-linear temporal interactions (illumination-invariant
    cues, direction of change, magnitude weighting).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(_cbr(4 * channels, channels, 1), _cbr(channels, channels, 3))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diff = x1 - x2
        cat = torch.cat([x1, x2, diff, torch.abs(diff)], dim=1)
        return diff + self.fuse(cat)


class TemporalInteraction(nn.Module):
    """Per-scale temporal module.

    mode='difference': DifferenceFusion only.
    mode='attention':  additionally re-weights the difference map with a learned
                       squeeze-excite channel gate (learned temporal attention).
    """

    def __init__(self, channels: int, mode: str = "difference"):
        super().__init__()
        assert mode in ("difference", "attention")
        self.mode = mode
        self.diff = DifferenceFusion(channels)
        if mode == "attention":
            hidden = max(channels // 4, 8)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        d = self.diff(x1, x2)
        if self.mode == "attention":
            d = d + d * self.gate(d)
        return d


class ConcatFusion(nn.Module):
    """Baseline: concat + 1x1 reduce."""

    def __init__(self, channels: int, **_: object):
        super().__init__()
        self.reduce = _cbr(2 * channels, channels, 1)

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        return self.reduce(torch.cat([opt, sar], dim=1))


class GatedFusion(nn.Module):
    """Learned per-pixel, per-channel gated blend of the two modalities.

    gate g = sigmoid(conv([proj(opt), proj(sar)])); out = g*opt' + (1-g)*sar'.
    Lets the network locally trust whichever modality is more reliable
    (e.g. SAR under clouds, optical in clear conditions).
    """

    def __init__(self, channels: int, **_: object):
        super().__init__()
        self.proj_o = _cbr(channels, channels, 1)
        self.proj_s = _cbr(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(2 * channels, channels, 1), nn.Sigmoid())

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        o, s = self.proj_o(opt), self.proj_s(sar)
        g = self.gate(torch.cat([o, s], dim=1))
        return g * o + (1.0 - g) * s


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between optical and SAR tokens.

    Spatial positions are tokens; optical tokens attend to SAR tokens and vice
    versa, so each modality can pull complementary evidence from the other at
    the *same* (aligned) or neighbouring locations. Pre-norm + FFN, residual
    throughout. Cost is O((HW)^2) per scale, cheap at strides >= 8; at stride 4
    with 256px inputs there are 4096 tokens which is still fine on a modern GPU.
    """

    def __init__(self, channels: int, num_heads: int = 4, max_tokens: int = 4096):
        super().__init__()
        self.max_tokens = max_tokens
        self.attn_o = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.attn_s = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm_o = nn.LayerNorm(channels)
        self.norm_s = nn.LayerNorm(channels)
        self.ffn_o = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 2 * channels), nn.GELU(),
            nn.Linear(2 * channels, channels),
        )
        self.ffn_s = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 2 * channels), nn.GELU(),
            nn.Linear(2 * channels, channels),
        )
        self.merge = _cbr(2 * channels, channels, 1)

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        B, C, H, W = opt.shape
        assert H * W <= self.max_tokens, (
            f"cross-attention scale has {H * W} tokens (> {self.max_tokens}); "
            "use out_strides with a smaller shallow stage, e.g. (8, 16, 32)."
        )
        o = opt.flatten(2).transpose(1, 2)  # [B, HW, C]
        s = sar.flatten(2).transpose(1, 2)
        o_attn, _ = self.attn_o(self.norm_o(o), self.norm_s(s), self.norm_s(s))
        s_attn, _ = self.attn_s(self.norm_s(s), self.norm_o(o), self.norm_o(o))
        o = o + o_attn
        s = s + s_attn
        o = o + self.ffn_o(o)
        s = s + self.ffn_s(s)
        o = o.transpose(1, 2).reshape(B, C, H, W)
        s = s.transpose(1, 2).reshape(B, C, H, W)
        return self.merge(torch.cat([opt + o, sar + s], dim=1))


def build_fusion(mode: str, channels: int, num_heads: int = 4) -> nn.Module:
    mode = (mode or "cross_attention").lower()
    if mode == "cross_attention":
        return CrossAttentionFusion(channels, num_heads=num_heads)
    if mode == "gated":
        return GatedFusion(channels)
    if mode == "concat":
        return ConcatFusion(channels)
    raise ValueError(f"unknown fusion mode: {mode}")
