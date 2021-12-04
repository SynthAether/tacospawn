from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsampler(nn.Module):
    """Gaussian upsampler from Non-attentive Tacotron.
    """
    def __init__(self, channels: int, layers: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            layers: the number of the BiGRUs.
        """
        super().__init__()
        self.bigrus = nn.ModuleList([
            nn.GRU(channels * 2, channels, batch_first=True, bidirectional=True)
            for _ in range(layers)])
        self.proj = nn.Linear(channels * 2, 2)

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Upsampling inputs w.r.t. predicted durations.
        Args:
            inputs: [torch.float32; [B, S, C]], input tensor.
            mask: [torch.float32; [B, S]], binary sequence mask.
            lengths: [torch.long; [B]], target spectrogram lengths, if provided.
        Returns:
            upsampled: [torch.float32; [B, T, C]], upsampled feature map.
            align: [torch.float32; [B, T, S]], alignment.
            lengths: [torch.long; [B]], spectrogram lengths.
            factor: [torch.float32; [B]], residual lengths.
        """
        x = inputs
        for bigru in self.bigrus:
            # [B, S, C]
            x, _ = bigru(x)
        # [B, S, 1], [B, S, 1]
        logdur, range_ = self.proj(x).chunk(2, dim=-1)
        # [B, S], [B, S]
        logdur, range_ = logdur.squeeze(dim=-1), range_.squeeze(dim=-1)
        # re-ranging
        if lengths is not None:
            # [B]
            factor = torch.log(lengths) - torch.logsumexp(
                logdur.masked_fill(~mask.to(torch.bool), -np.inf), dim=-1)
            # [B, S]
            logdur = logdur + factor[:, None]
        else:
            factor = None
        # [B, S], [B, S], masking
        dur, range_ = torch.exp(logdur) * mask, F.softplus(range_) * mask
        # [B]
        lengths = lengths or dur.sum(dim=-1)
        # [B, S]
        centers = torch.cumsum(dur) - 0.5 * dur
        # [T]
        timesteps = torch.arange(
            lengths.max(), dtype=torch.float32, device=centers.device)
        # [B, T]
        mel_mask = (timesteps[None] < lengths[:, None]).to(torch.float32)
        # [B, T, S]
        attn_mask = mel_mask[..., None] * mask[:, None]
        # [B, T, S]
        align = torch.square(
            (timesteps[None, :, None] - centers[:, None]) / range_[:, None])
        # [B, T, S]
        align = align / (
            (align * mask[:, None]).sum(dim=-1, keepdim=True) + 1e-5) * attn_mask
        # [B, T, C]
        upsampled = torch.matmul(align, inputs)
        # [B, T, S], [B], [B]
        return upsampled, align, lengths, factor
