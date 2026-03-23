"""Building blocks for FC-DenseNet (Tiramisu) and SkyScapesNet.

Implements both:
- Original Tiramisu blocks (DenseLayer, DenseBlock, TransitionDown, TransitionUp)
- SkyScapesNet blocks (SeparableConv2d, SeparableLayer, FullyDenseBlock, DoS, UpS)

Reference:
- "The One Hundred Layers Tiramisu" (Jégou et al., 2017)
- "SkyScapes — Fine-Grained Semantic Understanding of Aerial Scenes" (Azimi et al., ICCV 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared primitives
# =============================================================================

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise conv + pointwise conv."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# =============================================================================
# Original Tiramisu blocks (kept for FC-DenseNet-103 baseline)
# =============================================================================

class DenseLayer(nn.Module):
    """Original Tiramisu layer: BN -> ReLU -> Conv3x3 -> Dropout."""

    def __init__(self, in_channels, growth_rate, dropout_p=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x):
        return self.layer(x)


class DenseBlock(nn.Module):
    """Original Tiramisu Dense Block with dense connectivity.

    Returns (all_features, new_features_only).
    """

    def __init__(self, n_layers, in_channels, growth_rate, dropout_p=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, dropout_p)
            )

    def forward(self, x):
        new_features = []
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
            new_features.append(out)
        new_features_only = torch.cat(new_features, dim=1)
        return x, new_features_only


class TransitionDown(nn.Module):
    """Original Tiramisu Transition Down: BN -> ReLU -> Conv1x1 -> Dropout -> MaxPool2x2."""

    def __init__(self, in_channels, dropout_p=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Dropout2d(p=dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.layer(x)


class TransitionUp(nn.Module):
    """Original Tiramisu Transition Up: Transposed Convolution + skip concatenation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,
        )

    def forward(self, x, skip):
        out = self.conv_transpose(x)
        if out.shape[2:] != skip.shape[2:]:
            out = out[:, :, :skip.shape[2], :skip.shape[3]]
        return torch.cat([out, skip], dim=1)


# =============================================================================
# SkyScapesNet blocks (Fig 4 of the paper)
# =============================================================================

class SeparableLayer(nn.Module):
    """SL (Separable Layer) — replaces DenseLayer in SkyScapesNet.

    BN → ReLU → Conv3x3 → SeparableConv3x3 → Dropout
    Returns only NEW feature maps.
    """

    def __init__(self, in_channels, growth_rate, dropout_p=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            SeparableConv2d(growth_rate, growth_rate, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x):
        return self.layer(x)


class FullyDenseBlock(nn.Module):
    """FDB (Fully Dense Block) — replaces DenseBlock in SkyScapesNet.

    Uses SeparableLayers with dense connectivity PLUS additional residual
    connections between non-adjacent layers (inspired by DenseASPP).

    Each SL receives concatenated features from ALL previous SLs in this block,
    and additionally adds residual connections from non-adjacent layers.

    Returns (all_features, new_features_only).
    """

    def __init__(self, n_layers, in_channels, growth_rate, dropout_p=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.growth_rate = growth_rate

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                SeparableLayer(in_channels + i * growth_rate, growth_rate, dropout_p)
            )

        # Residual projection for non-adjacent connections:
        # Each layer gets a 1x1 conv to project accumulated features for residual add
        if n_layers > 1:
            self.residual_projections = nn.ModuleList()
            for i in range(1, n_layers):
                # Project from growth_rate * i channels to growth_rate
                self.residual_projections.append(
                    nn.Conv2d(growth_rate * i, growth_rate, kernel_size=1, bias=False)
                )
        else:
            self.residual_projections = None

    def forward(self, x):
        new_features = []
        for i, layer in enumerate(self.layers):
            out = layer(x)

            # Add residual from all previous new features (non-adjacent connections)
            if i > 0 and self.residual_projections is not None:
                prev_concat = torch.cat(new_features, dim=1)
                residual = self.residual_projections[i - 1](prev_concat)
                out = out + residual

            x = torch.cat([x, out], dim=1)
            new_features.append(out)

        new_features_only = torch.cat(new_features, dim=1)
        return x, new_features_only


class DownsamplingBlock(nn.Module):
    """DoS (Downsampling Block) — replaces TransitionDown in SkyScapesNet.

    BN → ReLU → Conv1x1 → SeparableConv3x3 → MaxPool2x2
    """

    def __init__(self, in_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            SeparableConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.layer(x)


class UpsamplingBlock(nn.Module):
    """UpS (Upsampling Block) — replaces TransitionUp in SkyScapesNet.

    TransposedConv3x3 for learned upsampling + UpS-NN (Nearest-Neighbor ×2
    with 1x1 projection) added together. Then concatenated with skip connection.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,
        )
        # UpS-NN path: project to out_channels before adding
        self.nn_project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False,
        )

    def forward(self, x, skip):
        # Learned upsampling via transposed convolution
        transposed = self.conv_transpose(x)

        # UpS-NN: nearest-neighbor ×2 upsampling + 1x1 projection
        nn_up = self.nn_project(F.interpolate(x, scale_factor=2, mode="nearest"))

        # Add the two upsampling paths
        out = transposed + nn_up[:, :, :transposed.shape[2], :transposed.shape[3]]

        # Handle spatial mismatch with skip
        if out.shape[2:] != skip.shape[2:]:
            out = out[:, :, :skip.shape[2], :skip.shape[3]]

        return torch.cat([out, skip], dim=1)
