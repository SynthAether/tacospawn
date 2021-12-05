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
    def __init__(self, factor: int):
        """Initializer.
        Args:
            factor: reduction factor.
        """
        super().__init__()
        self.factor = factor

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
            inputs = F.pad(inputs, [0, 0, 0, remains])
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


class PositionalEncodings(nn.Module):
    """Positional encodings from Vaswani et al., 2017.
    """
    def __init__(self, channels: int, size: int):
        """Initializer.
        Args:
            channels: size of the embeddings.
            size: size of the initial cache.
        """
        super().__init__()
        self.channels = channels
        self.size = size
        # caching
        self.register_buffer('cache', self.generate(size))

    def _load_from_state_dict(self):
        """Override load_state_dict for preventing cache load.
        """
        pass

    def forward(self, size: int) -> torch.Tensor:
        """Return cached positional encodings.
        Args:
            size: length of the pe.
        Returns:
            [torch.float32; [T, C]], sinusoidal positional encodings.
        """
        if size <= self.size:
            return self.cache[:size]
        # generate new cache
        self.size = size
        self.register_buffer('cache', self.generate(size))
        return self.cache

    def generate(self, size: int) -> torch.Tensor:
        """Generate positional encodings.
        Args:
            size: length of the pe.
        Returns:
           [torch.float32; [T, C]], sinusoidal positional encodings.
        """
        with torch.no_grad():
            # [T]
            pos = torch.arange(size)
            # [C // 2]
            i = torch.arange(0, self.channels, 2)
            # [C // 2]
            denom = torch.exp(-np.log(10000) * i / self.channels)
            # [T, C//2]
            context = pos[:, None] * denom[None]
            # [T, C//2, 2]
            pe = torch.stack([torch.sin(context), torch.cos(context)], dim=-1)
            # [T, C]
            pe = pe.view(size, self.channels)
        return pe
