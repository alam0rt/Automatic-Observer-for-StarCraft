"""Heatmap network: a small U-Net predicting a distribution over map cells.

Replaces the original Mask R-CNN. The camera target is "where on the map", which is
dense prediction rather than instance detection, so there are no anchors, boxes or
masks, and no COCO weights to throw away by swapping the first conv.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class InterestNet(nn.Module):
    def __init__(self, in_channels: int, width: int = 32, depth: int = 3):
        super().__init__()
        widths = [width * 2**i for i in range(depth + 1)]
        self.stem = block(in_channels, widths[0])
        self.down = nn.ModuleList(block(widths[i], widths[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) features -> (B, H, W) logits over cells."""
        skips = [self.stem(x)]
        for down in self.down:
            skips.append(down(F.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        for up in self.up:
            skip = skips.pop()
            y = up(torch.cat([F.interpolate(y, size=skip.shape[-2:], mode="nearest"), skip], dim=1))
        return self.head(y).squeeze(1)


def log_distribution(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.flatten(1), dim=1).view_as(logits)


def interest_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted cross-entropy between the target distribution and the predicted one."""
    per_item = -(target * log_distribution(logits)).flatten(1).sum(dim=1)
    return (per_item * weight).sum() / weight.sum()
