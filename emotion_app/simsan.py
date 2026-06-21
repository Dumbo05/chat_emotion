from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def gradient_reverse(values: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(values, strength)


class SqueezeExcitation2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.net(values)


class SeparableResidual2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1), dilation=(1, 1)):
        super().__init__()
        padding = tuple(value for value in dilation)
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 3, stride=stride, padding=padding,
                dilation=dilation, groups=in_channels, bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            SqueezeExcitation2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == (1, 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(values) + self.skip(values))


class TemporalResidual(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels, 3, padding=dilation,
                dilation=dilation, groups=channels, bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),
            nn.Tanh(),
            nn.Conv1d(channels // 2, 1, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(values), dim=-1)
        mean = torch.sum(values * weights, dim=-1)
        variance = torch.sum((values - mean.unsqueeze(-1)).square() * weights, dim=-1)
        return torch.cat([mean, torch.sqrt(variance.clamp_min(1e-5))], dim=1)


class SIMSAN(nn.Module):
    """Speaker-Invariant Multi-Scale Spectro-Temporal Attention Network."""

    def __init__(self, emotion_classes: int = 7, speaker_classes: int = 71, dropout: float = 0.25):
        super().__init__()
        # Two views: utterance-global normalization retains spectral shape,
        # frequency-wise CMVN suppresses stationary speaker/channel traits.
        self.stem_3 = nn.Sequential(
            nn.Conv2d(2, 16, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16), nn.SiLU(),
        )
        self.stem_5 = nn.Sequential(
            nn.Conv2d(2, 16, (5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(16), nn.SiLU(),
        )
        self.stem_7 = nn.Sequential(
            nn.Conv2d(2, 16, (7, 3), padding=(3, 1), bias=False),
            nn.BatchNorm2d(16), nn.SiLU(),
        )
        self.spectral = nn.Sequential(
            SeparableResidual2d(48, 64, stride=(2, 2)),
            SeparableResidual2d(64, 96, stride=(2, 2)),
            SeparableResidual2d(96, 128, stride=(2, 1)),
            SeparableResidual2d(128, 160, stride=(1, 2)),
        )
        self.temporal = nn.Sequential(
            TemporalResidual(160, 1, dropout),
            TemporalResidual(160, 2, dropout),
            TemporalResidual(160, 4, dropout),
            TemporalResidual(160, 8, dropout),
        )
        self.pool = AttentiveStatisticsPooling(160)
        self.embedding = nn.Sequential(
            nn.Linear(320, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(192, emotion_classes)
        self.speaker_head = nn.Sequential(
            nn.Linear(192, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, speaker_classes),
        )

    @staticmethod
    def normalized_views(log_mel: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, mel, time].
        global_mean = log_mel.mean(dim=(1, 2), keepdim=True)
        global_std = log_mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
        global_view = (log_mel - global_mean) / global_std

        frequency_mean = log_mel.mean(dim=2, keepdim=True)
        frequency_std = log_mel.std(dim=2, keepdim=True).clamp_min(1e-4)
        speaker_reduced = (log_mel - frequency_mean) / frequency_std
        return torch.stack([global_view, speaker_reduced], dim=1)

    def encode(self, log_mel: torch.Tensor) -> torch.Tensor:
        values = self.normalized_views(log_mel)
        values = torch.cat([
            self.stem_3(values), self.stem_5(values), self.stem_7(values)
        ], dim=1)
        values = self.spectral(values).mean(dim=2)
        values = self.temporal(values)
        return self.embedding(self.pool(values))

    def forward(self, log_mel: torch.Tensor, grl_strength: float = 0.0):
        embedding = self.encode(log_mel)
        emotion_logits = self.emotion_head(embedding)
        speaker_logits = self.speaker_head(
            gradient_reverse(embedding, grl_strength)
        )
        return emotion_logits, speaker_logits


class SIMSANInference(nn.Module):
    def __init__(self, model: SIMSAN):
        super().__init__()
        self.model = model

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        embedding = self.model.encode(log_mel)
        return self.model.emotion_head(embedding)
