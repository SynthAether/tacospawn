from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import Prenet


class Aligner(nn.Module):
    """Stepwise monotonic attention.
    """
    def __init__(self, inputs: int, channels: int):
        """Initializer.
        Args:
            input: size of the input tensors.
            channels: size of the internal hidden states.
        """
        super().__init__()
        self.attn = nn.GRUCell(inputs, channels)
        self.trans = nn.Sequential(
            nn.Linear(channels, channels), nn.Tanh(),
            nn.Linear(channels, 2), nn.Softmax(dim=-1))

    def state_init(self, encodings: torch.Tensor, mask: torch.Tensor) -> \
            Dict[str, torch.Tensor]:
        """Initialize states.
        Args:
            encodings: [torch.float32; [B, S, I]], text encodings.
            mask: [torch.float32; [B, S]], text mask.
        Returns:
            initial states.
        """
        with torch.no_grad():
            # B, S, _
            bsize, seqlen, _ = encodings.shape
            # [B, C]
            state = torch.zeros(bsize, self.attn.hidden_size, device=encodings.device)
            # [B, S]
            alpha = torch.zeros(bsize, seqlen, device=encodings.device)
            alpha[:, 0] = 1.
        return {'enc': encodings, 'mask': mask, 'state': state, 'alpha': alpha}

    def decode(self, frame: torch.Tensor, state: Dict[str, torch.Tensor]) -> \
            Dict[str, torch.Tensor]:
        """Compute align.
        Args:
            frame: [torch.float32; [B, H]], previous frame.
            state: state tensors.
        Returns:
            state: updated states.
        """
        # [B, I]
        prev = (state['encodings'] * state['alpha'][..., None]).sum(dim=1)
        # [B, C]
        state = self.attn(torch.cat([frame, prev], dim=-1), state['state'])
        # [B, 1]
        stop, next_ = self.trans(state).chunk(2, dim=-1)
        # [B, S]
        alpha = stop * state['alpha'] + next_ * F.pad(state['alpha'], [1, -1])
        return {**state, 'state': state, 'alpha': alpha * state['mask']}

    def forward(self,
                encodings: torch.Tensor,
                mask: torch.Tensor,
                gt: torch.Tensor) -> torch.Tensor:
        """Compute alignment between text encodings and ground-truth mel spectrogram.
        Args:
            encodings: [torch.float32; [B, S, I]], text encodings.
            mask: [torch.float32; [B, S]], text masks.
            gt: [torch.float32; [B, T, H]], preprocessed mel-spectrogram.
        Returns:
            alpha: [torch.float32; [B, T, S]], attention alignment.
        """
        state = self.state_init(encodings, mask)
        # T x [B, S]
        alphas = []
        # T x [B, H]
        for frame in gt.transpose(0, 1):
            state = self.decode(frame, state)
            alphas.append(state['alpha'])
        # [B, T, S]
        return torch.stack(alphas, dim=1)
