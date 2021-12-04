import itertools
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbhg import Cbhg
from .config import Config
from .decoder import Decoder
from .misc import PositionalEncodings, Prenet, Reduction
from .upsampler import Upsampler


class NonAttentiveTacotron(nn.Module):
    """Non-attentive tacotron for multispeakers
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configurations.
        """
        super().__init__()
        self.embedding = nn.Embedding(config.vocabs, config.embeddings)
        self.prenet = Prenet(
            config.embeddings,
            config.enc_prenet + [config.channels // 2],
            config.enc_dropout)

        self.cbhg = Cbhg(
            config.channels // 2,
            config.cbhg_banks,
            config.cbhg_pool,
            config.cbhg_kernels,
            config.cbhg_highways)

        self.reduction = Reduction(config.reduction)

        self.upsampler = Upsampler(
            config.channels + config.spkembed,
            config.channels // 2,
            config.upsampler_layers)

        self.pe = PositionalEncodings(self.pe, 30)

        self.decoder = Decoder(
            config.channels + config.spkembed + config.pe,
            config.channels,
            config.dec_prenet,
            config.dec_dropout,
            config.dec_layers,
            config.reduction * config.mel)

    def forward(self,
                inputs: torch.Tensor,
                textlen: torch.Tensor,
                spkembed: torch.Tensor,
                mel: Optional[torch.Tensor] = None,
                mellen: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode text tokens.
        Args;
            inputs: [torch.long; [B, S]], text symbol sequences.
            textlen: [torch.long; [B]], sequence lengths.
            spkembed: [torch.float32; [B, E]], speaker embeddings.
            mel: [torch.float32; [B, T, M]], mel-spectrogram, if provided.
            mellen: [torch.long; [B]], spectrogram lengths, if provided.
        Returns:
            mel: [torch.float32; [B, T, B]], predicted spectrogram.
            mellen: [torch.long; [B]], spectrogram lengths.
            auxiliary: auxiliary informations.
                align: [torch.float32; [B, T // F, S]], attention alignments.
                durations: [torch.float32; [B, S]], durations.
                factor: [torch.float32; [B]], size ratio between ground-truth and predicted lengths.
        """
        ## 1. Text encoding
        # S
        seqlen = inputs.size(1)
        # [B, S]
        text_mask = (
            torch.arange(seqlen, device=inputs.device)[None]
            < textlen[:, None]).to(torch.float32)
        # [B, S, E]
        embed = self.embedding(inputs)
        # [B, S, C // 2], masking for initial convolution of CBHG.
        preproc = self.prenet(embed) * text_mask[..., None]
        # [B, S, C]
        encodings = self.cbhg(preproc)
        # [B, S, C + E]
        encodings = torch.cat([
            encodings, spkembed[:, None].repeat(1, seqlen, 1)], dim=-1)

        ## 3. Decoding
        if mel is not None:
            # [B, T // F, F x M]
            mel, remains = self.reduction(mel)
            # [B], assume mellen is not None
            mellen = torch.ceil(mellen / self.reduction.factor).to(torch.long)
        else:
            remains = None
        # [B, T // F, C + E], [B], _
        upsampled, predlen, aux = self.upsampler(encodings, text_mask, mellen)
        # [B, T // F, P]
        pe = self.tokenwise_posenc(aux['durations'], upsampled.size(1))
        # [B, T // F, F x M]
        mel = self.decoder(torch.cat([upsampled, pe], dim=-1), mel)

        ## 4. Unfold
        if mellen is None:
            mellen = predlen.to(torch.long) * self.reduction.factor
        # [B, T, M]
        mel = self.reduction.unfold(mel, remains)
        # [B, T]
        mel_mask = (
            torch.arange(mel.size(1), device=mel.device)[None]
            < mellen[:, None]).to(torch.float32)
        # [B, T, M]
        return mel * mel_mask[..., None], mellen, aux

    def tokenwise_posenc(self, durations: torch.Tensor, maxlen: int) -> torch.Tensor:
        """Generate tokenwise positional encodings.
        Args:
            durations: [torch.float32; [B, S]], durations, not quantized.
            maxlen: total length of the posenc.
        Returns:
            [torch.float32; [B, T, C]], token-wise positional encodings.
        """
        # [B, S], quantize
        cumdur = torch.round(torch.cumsum(durations, dim=-1)).to(torch.long)
        # [B, S]
        durations = cumdur - F.pad(cumdur, [1, 0])
        # [K, C]
        cache = self.pe(durations.max())
        # [B, T, C]
        pe = torch.zeros(
            durations.size(0), maxlen, cache.size(-1), device=cache.device)
        # [], [S]
        for i, dur in enumerate(durations.detach().cpu().numpy()):
            # [T]
            indices = list(itertools.chain.from_iterable(range(s) for s in dur))
            # [T, C]
            pe[i, :len(indices)] = cache[indices]
        # [B, T, C]
        return pe
