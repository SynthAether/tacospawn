from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Prenet(nn.Sequential):
    """Bottleneck with dropout.
    """
    def __init__(self, channels: int, hiddens: List[int], dropout: float):
        """Initializer.
        Args:
            channels: size of the input channels.
            hiddens: size of the hidden channels.
            dropout: dropout rate.
        """
        super().__init__(*[
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Dropout(dropout))
            for in_channels, out_channels in zip([channels] + hiddens, hiddens)])


class Reduction(nn.Module):
    """Fold the inupts, applying reduction factor.
    """
    def __init__(self, factor: int, value: float = 0.):
        """Initializer.
        Args:
            factor: reduction factor.
            value: padding value.
        """
        super().__init__()
        self.factor = factor
        self.value = value

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Optional[int]]:
        """Fold the inputs, apply reduction factor.
        Args:
            input: [torch.float32; [B, T, C]], input tensor.
        Returns:
            [torch.float32; [B, T // F, F x C]] folded tensor and remains.
        """
        # B, T, C
        bsize, timesteps, channels = inputs.shape
        if timesteps % self.factor > 0:
            remains = self.factor - timesteps % self.factor
            # [B, T + R, C]
            inputs = F.pad(inputs, [0, 0, 0, remains], value=self.value)
        else:
            # no remains
            remains = None
        # [B, T // F, F x C]
        return inputs.reshape(bsize, -1, self.factor * channels), remains

    def unfold(self, inputs: torch.Tensor, remains: Optional[int]) -> torch.Tensor:
        """Recover the inputs, unfolding.
        Args:
            inputs: [torch.float32; [B, T // F, F x C]], folded tensor.
        Return:
            [torch.float32; [B, T, C]], recovered.
        """
        # B, _, F x C
        bsize, _, channels = inputs.shape
        # [B, T, C]
        recovered = inputs.reshape(bsize, -1, channels // self.factor)
        if remains is not None:
            # [B, T + R, C] -> [B, T, C]
            recovered = recovered[:, :-remains]
        return recovered
