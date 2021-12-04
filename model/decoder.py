from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import Prenet, Reduction


class Decoder(nn.Module):
    """GRU-based decoder.
    """
    def __init__(self,
                 channels: int,
                 hiddens: List[int],
                 dropout: float,
                 reduction: int,
                 layers: int,
                 mel: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            hiddnes: the size of the hidden units for decoder prenet.
            dropout: dropout rates for decoder prenet.
            reduction: reduction factor. 
            layers: the number of the GRU layers.
            mel: size of the output channels (channels of mel-spectrogram).
        """
        super().__init__()
        self.prenet = Prenet(reduction * mel, hiddens, dropout)
        self.reduction = Reduction(reduction)
        self.attn = nn.GRU(channels + hiddens[-1], channels, batch_first=True)
        self.grus = nn.ModuleList([
            nn.GRU(channels, channels, batch_first=True)
            for _ in range(layers)])
        self.proj = nn.Linear(channels, reduction * mel)

    def forward(self, inputs: torch.Tensor, gt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate spectrgram from intermediate features.
        Args:
            inputs: [torch.float32; [B, T // F, C]], input tensors.
            gt: [torch.float32; [B, T, M]], ground-truth spectrogram, if provided.
        Returns:
            [torch.float32; [B, T, M]], predicted spectrogram.
        """
        # autoregression
        if gt is None:
            return self.inference(inputs)
        # [B, T // F, F x M]
        gt, r = self.reduction(gt)
        # [B, T // F, H], pad for teacher force
        preproc = self.prenet(F.pad(gt, [0, 0, 1, -1]))
        # [B, T // F, C]
        x, _ = self.attn(torch.cat([inputs, preproc], dim=-1))
        for gru in self.grus:
            # [B, T // F, C]
            out, _ = gru(x)
            # [B, T // F, C], residual connection
            x = x + out
        # [B, T, M]
        return self.reduction.unfold(self.proj(x), r)

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate spectrogram autoregressively.
        Args:
            inputs: [torch.float32; [B, T // F, C]], input tensors.
        Returns:
            [torch.float32; [B, T, M]], predicted spectrogram.
        """
        # convert to cell
        attncell = self.grucell(self.attn)
        cells = [self.grucell(gru) for gru in self.grus]
        # B
        bsize = inputs.size(0)
        # [B, C], prepare hiddens
        attnhidd = torch.zeros(bsize, attncell.hidden_size, device=inputs.device)
        hiddens = [torch.zeros(bsize, gru.hidden_size) for gru in self.grus]
        # [B, F x M], start frame
        frame = torch.zeros(bsize, self.proj.out_features, device=inputs.device)
        # (T // F) x [B, F x M]
        frames = []
        # [B, C]
        for feat in inputs.transpose(0, 1):
            # [B, H]
            preproc = self.prenet(frame)
            # [B, C]
            attnhidd = attncell(torch.cat([feat, preproc], dim=-1), attnhidd)
            # [B, C], L x [B, C]
            x, new_hiddens = attnhidd, []
            for cell, hidden in zip(cells, hiddens):
                # [B, C]
                h = cell(x, hidden)
                # [B, C]
                x = x + h
                new_hiddens.append(h)
            # update hiddens
            hiddens = new_hiddens
            # [B, F x M]
            frame = self.proj(x)
            frames.append(frame)
        # [B, T // F, F x M]
        return self.reduction.unfold(torch.stack(frames, dim=1), remains=None)

    def grucell(self, gru: nn.GRU) -> nn.GRUCell:
        """Convert sequence operation to cell operation.
        Args:   
            gru: gated recurrent unit, sequence-level operation,
                assume unidirectional single-layer.
        Returns:
            GRU, cell-level operation.
        """
        cell = nn.GRUCell(gru.input_size, gru.hidden_size, bias=gru.bias)
        # weight copy
        cell.weight_ih = gru.weight_ih_l0
        cell.weight_hh = gru.weight_hh_l0
        if gru.bias:
            cell.bias_ih = gru.bias_ih_l0
            cell.bias_hh = gru.bias_hh_l0
        return cell
