from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsampler(nn.Module):
    """Gaussian upsampler from Non-attentive Tacotron.
    """
    def __init__(self, inputs: int, channels: int, layers: int):
        """Initializer.
        Args:
            inputs: size of the input channels.
            channels: size of the hidden channels.
            layers: the number of the BiGRUs.
        """
        super().__init__()
        self.proj_in = nn.Linear(inputs, channels * 2)
        self.bigrus = nn.ModuleList([
            nn.GRU(channels * 2, channels, batch_first=True, bidirectional=True)
            for _ in range(layers)])
        self.proj_out = nn.Linear(channels * 2, 2)

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Upsampling inputs w.r.t. predicted durations.
        Args:
            inputs: [torch.float32; [B, S, I]], input tensor.
            mask: [torch.float32; [B, S]], binary sequence mask.
            lengths: [torch.long; [B]], target spectrogram lengths, if provided.
        Returns:
            upsampled: [torch.float32; [B, T, I]], upsampled feature map.
            lengths: [torch.long; [B]], spectrogram lengths.
            auxiliaries: auxiliary outputs.
                align: [torch.float32; [B, T, S]], alignment.
                durations: [torch.float32; [B, S]], durations.
                factor: [torch.float32; [B]], residual lengths.
        """
        # [B, S, C x 2]
        x = self.proj_in(inputs)
        for bigru in self.bigrus:
            # [B, S, C x 2]
            x, _ = bigru(x)
        # [B, S, 1], [B, S, 1]
        logdur, range_ = self.proj_out(x).chunk(2, dim=-1)
        # [B, S], [B, S]
        logdur, range_ = logdur.squeeze(dim=-1), range_.squeeze(dim=-1)
        # re-ranging
        if lengths is not None:
            # [B]
            factor = torch.log(lengths.to(torch.float32)) - torch.logsumexp(
                logdur.masked_fill(~mask.to(torch.bool), -np.inf), dim=-1)
            # [B, S]
            logdur = logdur + factor[:, None]
        # [B, S], [B, S], masking
        dur, range_ = torch.exp(logdur) * mask, F.softplus(range_) * mask
        if lengths is None:
            factor = None
            # [B]
            lengths = dur.sum(dim=-1).to(torch.long)
        # [B, S]
        centers = torch.cumsum(dur, dim=-1) - 0.5 * dur
        # [T]
        timesteps = torch.arange(
            lengths.max(), dtype=torch.float32, device=centers.device)
        # [B, T]
        mel_mask = (timesteps[None] < lengths[:, None]).to(torch.float32)
        # [B, T, S]
        attn_mask = mel_mask[..., None] * mask[:, None]
        # [B, T, S]
        align = torch.square(
            (timesteps[None, :, None] - centers[:, None]) / (range_[:, None] + 1e-5))
        # [B, T, S]
        align = align / (
            (align * mask[:, None]).sum(dim=-1, keepdim=True) + 1e-5) * attn_mask
        # [B, T, I]
        upsampled = torch.matmul(align, inputs)
        # [B, T, I], [B], [B]
        return upsampled, lengths, {
            'align': align, 'durations': dur, 'factor': factor}
