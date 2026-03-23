"""SkyScapesNet: Multi-task model for fine-grained aerial scene understanding.

Architecture (Fig 3 of the paper):
- Modified FC-DenseNet-103 backbone with FDB (Fully Dense Blocks),
  separable convolutions, FRSR, CRASPP, and LKBR
- 5 downsampling steps, CRASPP at bottleneck
- 3 task branches split after the 2nd upsampling:
  1. Dense semantic segmentation (20 classes for SkyScapes-Dense)
  2. Multi-class edge detection (20 classes)
  3. Binary edge detection (1 channel)
- LKBR applied to skip connections for boundary refinement

Reference: "SkyScapes — Fine-Grained Semantic Understanding of Aerial Scenes"
(Azimi et al., ICCV 2019)
"""

import torch.nn as nn

from .layers import (
    FullyDenseBlock, DownsamplingBlock, UpsamplingBlock,
)
from .frsr import FRSR
from .craspp import CRASPP
from .lkbr import LKBR
from .hub import HubMixin


class DecoderBranch(nn.Module):
    """A single task-specific decoder branch (3 upsampling steps with FDBs + LKBR).

    Each branch receives features after the 2nd shared upsampling and continues
    with 3 more upsampling steps using its own FDBs, LKBRs, and skip connections.

    Args:
        n_layers_per_block: List of 3 layer counts for the 3 decoder FDBs.
        skip_channels: List of 3 skip connection channel counts (encoder order,
            i.e. skip3, skip2, skip1 from deepest to shallowest).
        growth_rate: Growth rate for FDBs.
        input_channels: Number of input channels (new features from previous stage).
        n_classes: Number of output classes.
        dropout_p: Dropout probability.
    """

    def __init__(self, n_layers_per_block, skip_channels, growth_rate,
                 input_channels, n_classes, dropout_p=0.2):
        super().__init__()
        assert len(n_layers_per_block) == 3
        assert len(skip_channels) == 3

        self.ups = nn.ModuleList()
        self.lkbrs = nn.ModuleList()
        self.fdbs = nn.ModuleList()

        upsample_channels = input_channels

        for i, n_layers in enumerate(n_layers_per_block):
            skip_ch = skip_channels[i]
            self.ups.append(UpsamplingBlock(upsample_channels, upsample_channels))
            self.lkbrs.append(LKBR(skip_ch))
            # After UpS concat with LKBR-processed skip (LKBR doubles channels via cat)
            concat_ch = upsample_channels + skip_ch * 2  # LKBR cats original + refined
            self.fdbs.append(
                FullyDenseBlock(n_layers, concat_ch, growth_rate, dropout_p)
            )
            upsample_channels = n_layers * growth_rate

        self.final_conv = nn.Conv2d(upsample_channels, n_classes, kernel_size=1)

    def forward(self, x, skips):
        """
        Args:
            x: Input features (new features from shared decoder).
            skips: List of 3 skip connections [skip3, skip2, skip1] (deepest to shallowest).
        """
        for ups, lkbr, fdb, skip in zip(self.ups, self.lkbrs, self.fdbs, skips):
            refined_skip = lkbr(skip)
            x = ups(x, refined_skip)
            _, x = fdb(x)  # keep only new features
        return self.final_conv(x)


class SkyScapesNet(nn.Module, HubMixin):
    """SkyScapesNet for fine-grained aerial semantic segmentation.

    Modified FC-DenseNet-103 with:
    - FDB (Fully Dense Blocks) with separable convolutions
    - FRSR (Full-Resolution Separable Residual) units
    - CRASPP (Concatenated Reverse ASPP) at bottleneck
    - LKBR (Large-Kernel Boundary Refinement) on skip connections
    - 3 task branches split after 2nd upsampling

    The encoder uses the same layer configuration as FC-DenseNet-103:
    [4, 5, 7, 10, 12] layers per block, growth_rate=32, init_features=48.

    Args:
        in_channels: Number of input image channels.
        n_classes: Number of segmentation classes (20 for SkyScapes-Dense).
        growth_rate: Growth rate for dense blocks.
        n_init_features: Initial convolution output channels.
        dropout_p: Dropout probability.
        craspp_mid_channels: CRASPP internal channel width.
    """

    _hub_config_keys = [
        "in_channels", "n_classes", "growth_rate",
        "n_init_features", "dropout_p", "craspp_mid_channels",
    ]

    def __init__(
        self,
        in_channels=3,
        n_classes=20,
        growth_rate=32,
        n_init_features=48,
        dropout_p=0.2,
        craspp_mid_channels=256,
    ):
        super().__init__()
        # Store config for HubMixin
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_init_features = n_init_features
        self.dropout_p = dropout_p
        self.craspp_mid_channels = craspp_mid_channels

        # FC-DenseNet-103 configuration
        n_layers_encoder = [4, 5, 7, 10, 12]
        n_layers_bottleneck = 15
        n_layers_decoder = [12, 10, 7, 5, 4]  # mirror of encoder

        self.growth_rate = growth_rate

        # --- Initial Convolution ---
        self.initial_conv = nn.Conv2d(
            in_channels, n_init_features, kernel_size=3, padding=1, bias=False,
        )

        # --- Encoder Path (5 FDBs + FRSR + DoS) ---
        self.encoder_fdbs = nn.ModuleList()
        self.encoder_frsrs = nn.ModuleList()
        self.encoder_dos = nn.ModuleList()

        current_channels = n_init_features
        skip_channels_list = []  # track channels at each skip point

        for n_layers in n_layers_encoder:
            fdb = FullyDenseBlock(n_layers, current_channels, growth_rate, dropout_p)
            current_channels = current_channels + n_layers * growth_rate
            self.encoder_fdbs.append(fdb)

            frsr = FRSR(current_channels)
            self.encoder_frsrs.append(frsr)

            skip_channels_list.append(current_channels)
            self.encoder_dos.append(DownsamplingBlock(current_channels))

        # --- Bottleneck: FDB + CRASPP ---
        self.bottleneck_fdb = FullyDenseBlock(
            n_layers_bottleneck, current_channels, growth_rate, dropout_p,
        )
        bottleneck_all_ch = current_channels + n_layers_bottleneck * growth_rate
        bottleneck_new_ch = n_layers_bottleneck * growth_rate  # 240

        self.craspp = CRASPP(
            in_channels=bottleneck_all_ch,
            out_channels=bottleneck_new_ch,
            mid_channels=craspp_mid_channels,
            dropout_p=dropout_p,
        )

        # --- Shared Decoder (first 2 upsampling steps, before branching) ---
        # These 2 steps use the deepest 2 skip connections (skip5, skip4)
        self.shared_ups = nn.ModuleList()
        self.shared_lkbrs = nn.ModuleList()
        self.shared_fdbs = nn.ModuleList()

        upsample_channels = bottleneck_new_ch  # 240 from CRASPP
        n_shared_decoder_steps = 2

        for i in range(n_shared_decoder_steps):
            skip_ch = skip_channels_list[-(i + 1)]  # skip5, skip4
            self.shared_ups.append(
                UpsamplingBlock(upsample_channels, upsample_channels)
            )
            self.shared_lkbrs.append(LKBR(skip_ch))
            # LKBR doubles skip channels via cat (original + refined)
            concat_ch = upsample_channels + skip_ch * 2
            n_layers = n_layers_decoder[i]
            self.shared_fdbs.append(
                FullyDenseBlock(n_layers, concat_ch, growth_rate, dropout_p)
            )
            upsample_channels = n_layers * growth_rate

        # After 2 shared decoder steps, upsample_channels = decoder[1] * growth_rate
        # = 10 * 16 = 160
        branch_input_ch = upsample_channels

        # Remaining skip connections for branches: skip3, skip2, skip1
        branch_skip_channels = [
            skip_channels_list[2],  # skip3
            skip_channels_list[1],  # skip2
            skip_channels_list[0],  # skip1
        ]
        branch_n_layers = n_layers_decoder[2:]  # [7, 5, 4]

        # --- Task Branches (each gets 3 remaining upsampling steps) ---

        # Branch 1: Semantic segmentation
        self.seg_branch = DecoderBranch(
            branch_n_layers, branch_skip_channels, growth_rate,
            branch_input_ch, n_classes, dropout_p,
        )

        # Branch 2: Multi-class edge detection
        self.multi_edge_branch = DecoderBranch(
            branch_n_layers, branch_skip_channels, growth_rate,
            branch_input_ch, n_classes, dropout_p,
        )

        # Branch 3: Binary edge detection
        self.binary_edge_branch = DecoderBranch(
            branch_n_layers, branch_skip_channels, growth_rate,
            branch_input_ch, 1, dropout_p,
        )

    def forward(self, x):
        # --- Initial Conv ---
        out = self.initial_conv(x)

        # --- Encoder ---
        skips = []
        for fdb, frsr, dos in zip(
            self.encoder_fdbs, self.encoder_frsrs, self.encoder_dos
        ):
            out, _ = fdb(out)        # FDB: all features
            out = frsr(out)          # FRSR: full-resolution residual
            skips.append(out)        # store skip connection
            out = dos(out)           # Downsample

        # --- Bottleneck ---
        out, _ = self.bottleneck_fdb(out)  # FDB: all features
        _, out = self.craspp(out)          # CRASPP: new features only

        # --- Shared Decoder (2 upsampling steps) ---
        for i, (ups, lkbr, fdb) in enumerate(
            zip(self.shared_ups, self.shared_lkbrs, self.shared_fdbs)
        ):
            skip = lkbr(skips[-(i + 1)])  # LKBR: refine skip5, skip4
            out = ups(out, skip)          # Upsample + concat with skip
            _, out = fdb(out)             # FDB: new features only

        # --- Branch Decoder (3 upsampling steps each) ---
        # Remaining skips: skip3, skip2, skip1
        branch_skips = [skips[2], skips[1], skips[0]]

        seg = self.seg_branch(out, branch_skips)
        multi_edge = self.multi_edge_branch(out, branch_skips)
        binary_edge = self.binary_edge_branch(out, branch_skips)

        return seg, multi_edge, binary_edge
