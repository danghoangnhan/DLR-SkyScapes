"""FC-DenseNet (Tiramisu) for semantic segmentation.

Implements the architecture from:
"The One Hundred Layers Tiramisu: Fully Convolutional DenseNets
for Semantic Segmentation" (Jégou et al., 2017)

Supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103 configurations.
"""

import torch.nn as nn

from .layers import DenseBlock, TransitionDown, TransitionUp
from .hub import HubMixin


class FCDenseNet(nn.Module, HubMixin):
    """Fully Convolutional DenseNet for semantic segmentation.

    Architecture: Initial Conv -> [DenseBlock + TransitionDown] * n_pools
                  -> Bottleneck DenseBlock
                  -> [TransitionUp + skip + DenseBlock] * n_pools
                  -> Final Conv1x1

    Args:
        in_channels: Number of input image channels (default: 3).
        n_classes: Number of output segmentation classes.
        growth_rate: Number of new feature maps per DenseLayer (k).
        n_init_features: Number of filters in the initial convolution.
        n_layers_per_block: Layer counts for encoder blocks + bottleneck + decoder blocks.
            For FC-DenseNet103: [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
        dropout_p: Dropout probability.
        bottleneck_module: Optional nn.Module to replace the default bottleneck DenseBlock.
            Must have an `out_channels` attribute indicating output channel count.
    """

    _hub_config_keys = [
        "in_channels", "n_classes", "growth_rate",
        "n_init_features", "n_layers_per_block", "dropout_p",
    ]

    def __init__(
        self,
        in_channels=3,
        n_classes=31,
        growth_rate=16,
        n_init_features=48,
        n_layers_per_block=None,
        dropout_p=0.2,
        bottleneck_module=None,
    ):
        super().__init__()

        if n_layers_per_block is None:
            # FC-DenseNet103 default
            n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

        # Store config for HubMixin
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_init_features = n_init_features
        self.n_layers_per_block = n_layers_per_block
        self.dropout_p = dropout_p

        assert len(n_layers_per_block) % 2 == 1, \
            "n_layers_per_block must have odd length (encoder + bottleneck + decoder)"

        n_pools = len(n_layers_per_block) // 2
        n_layers_encoder = n_layers_per_block[:n_pools]
        n_layers_bottleneck = n_layers_per_block[n_pools]
        n_layers_decoder = n_layers_per_block[n_pools + 1:]

        self.growth_rate = growth_rate

        # --- Initial Convolution ---
        self.initial_conv = nn.Conv2d(
            in_channels, n_init_features, kernel_size=3, padding=1, bias=False
        )

        # --- Encoder Path ---
        self.encoder_blocks = nn.ModuleList()
        self.transition_downs = nn.ModuleList()

        current_channels = n_init_features
        skip_channels = []

        for n_layers in n_layers_encoder:
            block = DenseBlock(n_layers, current_channels, growth_rate, dropout_p)
            current_channels = current_channels + n_layers * growth_rate
            self.encoder_blocks.append(block)
            skip_channels.append(current_channels)
            self.transition_downs.append(TransitionDown(current_channels, dropout_p))

        # --- Bottleneck ---
        if bottleneck_module is not None:
            self.bottleneck = bottleneck_module
            bottleneck_new_channels = bottleneck_module.out_channels
        else:
            self.bottleneck = DenseBlock(
                n_layers_bottleneck, current_channels, growth_rate, dropout_p
            )
            bottleneck_new_channels = n_layers_bottleneck * growth_rate

        # --- Decoder Path ---
        self.transition_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        upsample_channels = bottleneck_new_channels

        for i, n_layers in enumerate(n_layers_decoder):
            skip_ch = skip_channels[-(i + 1)]
            self.transition_ups.append(
                TransitionUp(upsample_channels, upsample_channels)
            )
            concat_channels = upsample_channels + skip_ch
            self.decoder_blocks.append(
                DenseBlock(n_layers, concat_channels, growth_rate, dropout_p)
            )
            # Only NEW features propagate to the next TransitionUp
            upsample_channels = n_layers * growth_rate

        # --- Final Classification ---
        self.n_features_out = upsample_channels  # exposed for backbone usage
        if n_classes is not None:
            self.final_conv = nn.Conv2d(upsample_channels, n_classes, kernel_size=1)
        else:
            self.final_conv = None

    def forward(self, x):
        # Initial convolution
        out = self.initial_conv(x)

        # Encoder: store skip connections
        skips = []
        for block, td in zip(self.encoder_blocks, self.transition_downs):
            out, _ = block(out)  # all_features for skip, discard new_only
            skips.append(out)
            out = td(out)

        # Bottleneck: keep only NEW features
        _, out = self.bottleneck(out)

        # Decoder: use skip connections in reverse
        for tu, block, skip in zip(
            self.transition_ups, self.decoder_blocks, reversed(skips)
        ):
            out = tu(out, skip)  # upsample + concatenate with skip
            _, out = block(out)  # keep only NEW features

        if self.final_conv is not None:
            return self.final_conv(out)
        return out

    @classmethod
    def densenet56(cls, in_channels=3, n_classes=31, **kwargs):
        return cls(
            in_channels=in_channels,
            n_classes=n_classes,
            n_layers_per_block=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            **kwargs,
        )

    @classmethod
    def densenet67(cls, in_channels=3, n_classes=31, **kwargs):
        return cls(
            in_channels=in_channels,
            n_classes=n_classes,
            n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
            growth_rate=16,
            n_init_features=48,
            **kwargs,
        )

    @classmethod
    def densenet103(cls, in_channels=3, n_classes=31, **kwargs):
        return cls(
            in_channels=in_channels,
            n_classes=n_classes,
            n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
            growth_rate=16,
            n_init_features=48,
            **kwargs,
        )
