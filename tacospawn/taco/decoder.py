from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import Prenet
from .align import Aligner


class Decoder(nn.Module):
    """GRU-based decoder.
    """
    def __init__(self,
                 inputs: int,
                 channels: int,
                 hiddens: List[int],
                 dropout: float,
                 layers: int,
                 mel: int,
                 max_factor: int):
        """Initializer.
        Args:
            inputs: size of the input channels.
            channels: size of the hidden channels.
            hiddens: the size of the hidden units for decoder prenet.
            dropout: dropout rates for decoder prenet.
            layers: the number of the GRU layers.
            mel: size of the output channels (channels of mel-spectrogram).
            max_factor: assume decoder max length as textlen x max_factor.
        """
        super().__init__()
        self.prenet = Prenet(mel, hiddens, dropout)
        self.aligner = Aligner(inputs + hiddens[-1], channels)
        self.blender = nn.GRU(inputs + hiddens[-1], channels)
        self.grus = nn.ModuleList([
            nn.GRU(channels, channels, batch_first=True)
            for _ in range(layers)])
        self.proj = nn.Linear(channels, mel)
        self.max_factor = max_factor

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.Tensor,
                gt: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate spectrgram from intermediate features.
        Args:
            inputs: [torch.float32; [B, S, I]], input tensors.
            mask: [torch.float32, [B, S]], text masks.
            gt: [torch.float32; [B, T, M]], ground-truth spectrogram, if provided.
        Returns:
            [torch.float32; [B, T, M]], predicted spectrogram.
        """
        # autoregression
        if gt is None:
            return self.inference(inputs, mask)
        # [B, T, H], pad for teacher force, default zero frame.
        preproc = self.prenet(F.pad(gt, [0, 0, 1, -1], value=np.log(1e-5)))
        # [B, T, S]
        alpha = self.aligner(inputs, mask, preproc)
        # [B, T, I]
        aligned = torch.matmul(alpha, inputs)
        # [B, T, C]
        x, _ = self.blender(torch.cat([aligned, preproc], dim=-1))
        for gru in self.grus:
            # [B, T, C]
            out, _ = gru(x)
            # [B, T, C], residual connection
            x = x + out
        # [B, T, M]
        return self.proj(x), {'align': alpha}

    def inference(self, inputs: torch.Tensor, mask: torch.Tensor) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate spectrogram autoregressively.
        Args:
            inputs: [torch.float32; [B, S, I]], input tensors.
            mask: [torch.float32; [B, S]], text masks.
        Returns:
            [torch.float32; [B, T, M]], predicted spectrogram.
        """
        state = self.alinger.state_init(inputs, mask)
        # convert to cell
        blender = self.grucell(self.blender)
        cells = [self.grucell(gru) for gru in self.grus]
        # B, S, _
        bsize, seqlen, _ = inputs.shape
        # [B, C], L x [B, C], prepare hiddens
        bhidden = torch.zeros(bsize, blender.hidden_size, device=inputs.device)
        chidden = [torch.zeros(bsize, gru.hidden_size, device=inputs.device)
                   for gru in self.grus]
        # [B]
        textlen = mask.sum(dim=-1).long()
        # [B, M], start frame
        frame = torch.full([bsize, self.proj.out_features], np.log(1e-5), device=inputs.device)
        # [B], initial spectrogram lengths.
        mellen = torch.zeros(bsize, device=inputs.device)
        # T x [B, M], T x [B, S]
        frames, alphas = [], []
        for timestep in range(seqlen * self.max_factor):
            # [B, H]
            preproc = self.prenet(frame)
            # compute align
            state = self.aligner.decode(preproc, state)
            alphas.append(state['alpha'])
            # [B, I]
            aligned = (inputs * state['alpha'][..., None]).sum(dim=1)
            # [B, C]
            bhidden = blender(torch.cat([aligned, preproc], dim=-1), bhidden)
            # [B, C], L x [B, C]
            x, new_hiddens = bhidden, []
            for cell, hidden in zip(cells, chidden):
                # [B, C]
                h = cell(x, hidden)
                # [B, C]
                x = x + h
                new_hiddens.append(h)
            # update hiddens
            chidden = new_hiddens
            # [B, M]
            frame = self.proj(x)
            frames.append(frame)
            # [B], check end of sequence.
            eos = torch.argmax(state['alpha'], dim=-1) >= textlen
            # B
            mellen = torch.where(eos, timestep, mellen)
            # if all done
            if eos.all():
                break
        # [B, T // F, F x M]
        return torch.stack(frames, dim=1), {
            'align': torch.stack(alphas, dim=1), 'mellen': mellen}

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
