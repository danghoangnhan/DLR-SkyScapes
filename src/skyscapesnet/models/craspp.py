"""Concatenated Reverse ASPP (CRASPP) for SkyScapesNet.

Unlike standard parallel ASPP, CRASPP uses two SEQUENTIAL cascading paths:
- Forward path: large→small dilation (18→12→6→1x1), each output concatenated
  before feeding the next stage
- Reverse path: small→large dilation (1x1→6→12→18), cascading similarly
- Final: concatenate both paths → Add

This captures receptive fields optimal for both small and large objects.

Reference: "SkyScapes — Fine-Grained Semantic Understanding of Aerial Scenes"
(Azimi et al., ICCV 2019), Section 4 and Figure 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRASPP(nn.Module):
    """Concatenated Reverse ASPP — cascading sequential design.

    Forward path (large→small dilation):
        ImagePooling → Cat → Atrous18 Conv3x3 → Cat → Atrous12 Conv3x3
        → Cat → Atrous6 Conv3x3 → Cat → Conv1x1

    Reverse path (small→large dilation):
        Conv1x1 → Cat → Atrous6 Conv3x3 → Cat → Atrous12 Conv3x3
        → Cat → Atrous18 Conv3x3

    Final: Cat(forward_out, reverse_out) → Add

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count (exposed for decoder).
        mid_channels: Internal channel width per stage.
        dropout_p: Dropout probability.
    """

    def __init__(self, in_channels, out_channels=240, mid_channels=256, dropout_p=0.2):
        super().__init__()
        self.out_channels = out_channels

        # --- Forward path: ImagePooling → Atrous18 → Atrous12 → Atrous6 → Conv1x1 ---

        # Image-level pooling (global context)
        # No BN after pool — spatial dims are 1x1 which breaks BN with batch_size=1
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Forward cascading atrous convolutions
        # After image_pool cat with input: in_channels + mid_channels
        self.fwd_atrous18 = self._make_atrous(in_channels + mid_channels, mid_channels, dilation=18)
        # After cat: in_channels + mid_channels + mid_channels
        self.fwd_atrous12 = self._make_atrous(in_channels + 2 * mid_channels, mid_channels, dilation=12)
        # After cat: in_channels + 3 * mid_channels
        self.fwd_atrous6 = self._make_atrous(in_channels + 3 * mid_channels, mid_channels, dilation=6)
        # After cat: in_channels + 4 * mid_channels
        self.fwd_conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels + 4 * mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # --- Reverse path: Conv1x1 → Atrous6 → Atrous12 → Atrous18 ---

        self.rev_conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # After cat: in_channels + mid_channels
        self.rev_atrous6 = self._make_atrous(in_channels + mid_channels, mid_channels, dilation=6)
        # After cat: in_channels + 2 * mid_channels
        self.rev_atrous12 = self._make_atrous(in_channels + 2 * mid_channels, mid_channels, dilation=12)
        # After cat: in_channels + 3 * mid_channels
        self.rev_atrous18 = self._make_atrous(in_channels + 3 * mid_channels, mid_channels, dilation=18)

        # Final projection: forward (5*mid) + reverse (4*mid) → out_channels
        # Forward produces: mid (pool) + mid (a18) + mid (a12) + mid (a6) + mid (1x1) = 5*mid
        # Reverse produces: mid (1x1) + mid (a6) + mid (a12) + mid (a18) = 4*mid
        self.final_project = nn.Sequential(
            nn.Conv2d(9 * mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    @staticmethod
    def _make_atrous(in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # --- Forward path ---
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=(h, w), mode="bilinear", align_corners=False)

        fwd = torch.cat([x, pool], dim=1)
        fwd_a18 = self.fwd_atrous18(fwd)

        fwd = torch.cat([fwd, fwd_a18], dim=1)
        fwd_a12 = self.fwd_atrous12(fwd)

        fwd = torch.cat([fwd, fwd_a12], dim=1)
        fwd_a6 = self.fwd_atrous6(fwd)

        fwd = torch.cat([fwd, fwd_a6], dim=1)
        fwd_1x1 = self.fwd_conv1x1(fwd)

        fwd_out = torch.cat([pool, fwd_a18, fwd_a12, fwd_a6, fwd_1x1], dim=1)

        # --- Reverse path ---
        rev_1x1 = self.rev_conv1x1(x)

        rev = torch.cat([x, rev_1x1], dim=1)
        rev_a6 = self.rev_atrous6(rev)

        rev = torch.cat([rev, rev_a6], dim=1)
        rev_a12 = self.rev_atrous12(rev)

        rev = torch.cat([rev, rev_a12], dim=1)
        rev_a18 = self.rev_atrous18(rev)

        rev_out = torch.cat([rev_1x1, rev_a6, rev_a12, rev_a18], dim=1)

        # --- Combine ---
        combined = torch.cat([fwd_out, rev_out], dim=1)
        out = self.final_project(combined)

        # Return (all, new) to match DenseBlock interface
        return out, out
