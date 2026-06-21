"""
NAVE model -- Normalized, Adaptive Conformer for Whale Vocalization-Event Detection
===================================================================================
Self-contained definition of the NAVE architecture. Composition:

    4-ch STFT+PCEN  ->  FDY stem (filterbank + feat0 frequency-dynamic)
                    ->  residual stack (bottleneck + depthwise)
                    ->  flatten (C*F) -> linear projection -> d_model
                    ->  N x Conformer block (macaron FFN / RoPE MHSA /
                        wide depthwise conv / macaron FFN)
                    ->  linear frame head -> per-frame logits (B, T, 3)

All hyperparameters are taken from ``nave_config``. The projection is built
eagerly at construction from a shape probe (the flattened CNN width is fixed by
the STFT settings), so the state_dict is complete before the first forward.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import nave_config as cfg


# ======================================================================
# Frequency-Dynamic 2-D convolution  (the "Adaptive" in NAVE)
# ======================================================================

class FDYConv2d(nn.Module):
    """Frequency-Dynamic Conv2d (Nam et al., Interspeech 2022): K basis kernels
    mixed per output frequency bin by a softmax attention over the time-averaged
    input. The attention branch uses the same frequency kernel/stride/padding as
    the main conv, so its output bin count matches even when strided."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size, stride=1,
                 padding=0, n_basis: int = 4, temperature: float = 1.0,
                 att_hidden: int | None = None, bias: bool = True):
        super().__init__()
        kf, kt = _pair(kernel_size)
        sf, st = _pair(stride)
        pf, pt = _pair(padding)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kf, self.kt = kf, kt
        self.sf, self.st = sf, st
        self.pf, self.pt = pf, pt
        self.n_basis = int(n_basis)
        self.temperature = float(temperature)

        self.weight = nn.Parameter(torch.empty(self.n_basis, out_ch, in_ch, kf, kt))
        self.bias = nn.Parameter(torch.zeros(self.n_basis, out_ch)) if bias else None
        self.reset_parameters()

        hidden = att_hidden or max(in_ch, self.n_basis * 4)
        self.att = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kf, stride=sf, padding=pf),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, self.n_basis, kernel_size=1),
        )

    def reset_parameters(self) -> None:
        for k in range(self.n_basis):
            nn.init.kaiming_uniform_(self.weight[k], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_ch * self.kf * self.kt
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, n_basis: int = 4,
                  temperature: float = 1.0) -> "FDYConv2d":
        if conv.groups != 1:
            raise ValueError("FDYConv2d only supports groups=1 convs")
        return cls(
            conv.in_channels, conv.out_channels,
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, n_basis=n_basis, temperature=temperature,
            bias=conv.bias is not None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, C, F, T)
        B, C, _, T = x.shape
        a = x.mean(dim=3)                                      # (B, C, F)
        a = self.att(a)                                        # (B, K, F')
        a = torch.softmax(a / self.temperature, dim=1)
        Fp = a.shape[-1]

        w = self.weight.reshape(self.n_basis * self.out_ch, self.in_ch,
                                self.kf, self.kt)
        b = self.bias.reshape(-1) if self.bias is not None else None
        y = F.conv2d(x, w, bias=b, stride=(self.sf, self.st),
                     padding=(self.pf, self.pt))               # (B, K*out, F', T')
        Tp = y.shape[-1]
        y = y.view(B, self.n_basis, self.out_ch, y.shape[-2], Tp)
        if y.shape[-2] != Fp:
            raise RuntimeError(
                f"FDY freq mismatch: conv F'={y.shape[-2]} vs attention F'={Fp}")
        a = a.view(B, self.n_basis, 1, Fp, 1)
        return (y * a).sum(dim=1)                              # (B, out, F', T')

    def extra_repr(self) -> str:
        return (f"in={self.in_ch}, out={self.out_ch}, "
                f"k=({self.kf},{self.kt}), s=({self.sf},{self.st}), "
                f"p=({self.pf},{self.pt}), K={self.n_basis}, T={self.temperature:g}")


# ======================================================================
# CNN stem (WhaleVAD feature extractor, FDY on the two early convs)
# ======================================================================

class ResidualBlock(nn.Module):
    """Sum-connected residual container: x -> x + block_1(x) -> ... ."""

    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class NAVEStem(nn.Module):
    """Phase-aware 4-ch input -> (B, C, F', T) features. ``filterbank`` and the
    first ``feat_extractor`` conv are frequency-dynamic (FDY); everything else
    follows the WhaleVAD CNN feature extractor."""

    def __init__(self):
        super().__init__()
        ch_in = cfg.FEAT_CHANNELS
        fb_ch = cfg.FILTERBANK_OUT_CH
        fe_ch = cfg.FEAT_EXTRACTOR_CH
        bn_ch = cfg.BOTTLENECK_CH

        # 1. Filterbank: (7,1) over frequency, stride 3 (freq 129 -> 41). FDY.
        self.filterbank = FDYConv2d.from_conv(
            nn.Conv2d(ch_in, fb_ch, kernel_size=(7, 1), stride=(3, 1), padding=0),
            n_basis=cfg.FDY_BASIS, temperature=cfg.FDY_TEMP)

        # 2. Feature extractor: first conv (5,5)/s(3,1) is FDY (freq 41 -> 14).
        self.feat_extractor = nn.Sequential(
            FDYConv2d.from_conv(
                nn.Conv2d(fb_ch, fe_ch, kernel_size=(5, 5), stride=(3, 1), padding=(2, 2)),
                n_basis=cfg.FDY_BASIS, temperature=cfg.FDY_TEMP),
            nn.BatchNorm2d(fe_ch),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(5, 1), stride=1, padding=0),
            nn.Conv2d(fe_ch, fe_ch, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(fe_ch),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=0),
        )

        # 3. Bottleneck (128 -> 64 -> 64 -> 128 squeeze/expand).
        bottleneck = nn.Sequential(
            nn.Conv2d(fe_ch, bn_ch, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.GELU(), nn.Dropout(cfg.BOTTLENECK_DROPOUT),
            nn.Conv2d(bn_ch, bn_ch, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GELU(), nn.Dropout(cfg.BOTTLENECK_DROPOUT),
            nn.Conv2d(bn_ch, fe_ch, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(fe_ch), nn.GELU(), nn.Dropout(cfg.BOTTLENECK_DROPOUT),
        )
        # 4. Depthwise aggregation (3 x depthwise 3x3, widening the time field).
        aggregation = nn.Sequential(
            nn.Dropout2d(cfg.AGG_DROPOUT),
            nn.Conv2d(fe_ch, fe_ch, (3, 3), (1, 1), 1, groups=fe_ch),
            nn.BatchNorm2d(fe_ch), nn.GELU(),
            nn.Conv2d(fe_ch, fe_ch, (3, 3), (1, 1), 1, groups=fe_ch),
            nn.BatchNorm2d(fe_ch), nn.GELU(),
            nn.Conv2d(fe_ch, fe_ch, (3, 3), (1, 1), 1, groups=fe_ch),
            nn.BatchNorm2d(fe_ch), nn.GELU(),
        )
        self.residual_stack = ResidualBlock(bottleneck, aggregation)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.filterbank(spec)
        x = self.feat_extractor(x)
        x = self.residual_stack(x)
        return x                                               # (B, C, F', T)


# ======================================================================
# Conformer encoder
# ======================================================================

def _rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = _rope_cache(T, self.head_dim, x.device, x.dtype)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask)[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class FeedForward(nn.Module):
    """Macaron feed-forward module (applied with a 1/2 residual weight)."""

    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        hidden = d_model * mult
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """LayerNorm -> pointwise(->2d) -> GLU -> depthwise(k) -> BN -> SiLU ->
    pointwise(->d) -> dropout. Operates on (B, T, d_model)."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1, "conv kernel must be odd for 'same' padding"
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size,
                            padding=(kernel_size - 1) // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x).transpose(1, 2)                       # (B, D, T)
        x = F.glu(self.pw1(x), dim=1)
        x = self.dw(x)
        x = self.act(self.bn(x))
        x = self.dropout(self.pw2(x))
        return x.transpose(1, 2)                               # (B, T, D)


class ConformerBlock(nn.Module):
    """macaron-FFN / RoPE-MHSA / conv / macaron-FFN, pre-norm residual."""

    def __init__(self, d_model, nhead, ffn_mult, conv_kernel, dropout):
        super().__init__()
        self.ffn1 = FeedForward(d_model, ffn_mult, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, nhead, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForward(d_model, ffn_mult, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn_drop(self.attn(self.attn_norm(x), key_padding_mask))
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


# ======================================================================
# NAVE
# ======================================================================

class NAVE(nn.Module):
    """Normalized, Adaptive Conformer for Whale Vocalization-Event Detection."""

    def __init__(self):
        super().__init__()
        self.stem = NAVEStem()
        self.d_model = cfg.D_MODEL

        # Eager projection: probe the (fixed) flattened CNN width in eval mode
        # (no BN-stat updates), then build proj so the state_dict is complete.
        was_training = self.stem.training
        self.stem.eval()
        with torch.no_grad():
            probe = torch.zeros(1, cfg.FEAT_CHANNELS, cfg.N_FFT // 2 + 1, 64)
            feat = self.stem(probe)
        self.stem.train(was_training)
        proj_in = feat.shape[1] * feat.shape[2]                # C * F'
        self.proj = nn.Linear(proj_in, self.d_model)
        self.input_drop = nn.Dropout(cfg.DROPOUT)

        self.blocks = nn.ModuleList([
            ConformerBlock(cfg.D_MODEL, cfg.NHEAD, cfg.FFN_MULT, cfg.CONV_KERNEL, cfg.DROPOUT)
            for _ in range(cfg.NUM_LAYERS)
        ])
        self.head = nn.Linear(cfg.D_MODEL, cfg.N_CLASSES)

    def forward(self, spec: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stem(spec)                                    # (B, C, F', T)
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fr)
        x = self.input_drop(self.proj(x))                      # (B, T, d_model)

        if key_padding_mask is not None and key_padding_mask.size(1) != T:
            kpm = key_padding_mask
            if kpm.size(1) > T:
                kpm = kpm[:, :T]
            else:
                pad = torch.ones(B, T - kpm.size(1), dtype=torch.bool, device=kpm.device)
                kpm = torch.cat([kpm, pad], dim=1)
            key_padding_mask = kpm

        for blk in self.blocks:
            x = blk(x, key_padding_mask)
        return self.head(x)                                    # (B, T, N_CLASSES)

    def load_checkpoint(self, path, map_location="cpu"):
        """Load a NAVE checkpoint (state_dict under 'model_state_dict' or raw)."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.load_state_dict(sd, strict=True)
        return ckpt if isinstance(ckpt, dict) else None
