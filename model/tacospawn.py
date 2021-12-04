from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .nat import NonAttentiveTacotron


class TacoSpawn(nn.Module):
    """TacoSpawn variations of Non-attentive Tacotron.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configurations.
        """
        super().__init__()
        self.nat = NonAttentiveTacotron(config)
        # unconditional prior space, mean and log stddev
        self.priorbuffer = nn.Parameter(
            torch.randn(config.modal, config.spkembed * 2 + 1),
            requires_grad=True)
        # speaker embedding
        self.spkembed = nn.Embedding(config.speakers, config.spkembed)

    def forward(self,
                text: torch.Tensor,
                textlen: torch.Tensor,
                mel: Optional[torch.Tensor] = None,
                mellen: Optional[torch.Tensor] = None,
                spkid: Optional[torch.Tensor] = None,
                modalid: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode text tokens.
        Args;
            text: [torch.long; [B, S]], text symbol sequences.
            textlen: [torch.long; [B]], sequence lengths.
            mel: [torch.float32; [B, T, M]], mel-spectrogram, if provided.
            mellen: [torch.long; [B]], spectrogram lengths, if provided.
            spkid: [torch.long; [B]], speaker id, if provided.
            modalid: [torch.long; [B]], modal id, if provided,
                if both spkid and modalid provided, return sample based on spkid.
        Returns:
            mel: [torch.float32; [B, T, B]], predicted spectrogram.
            mellen: [torch.long; [B]], spectrogram lengths.
            auxiliary: auxiliary informations.
                spkembed: [torch.float32; [B, E]], sampled speaker embedding.
                align: [torch.float32; [B, T // F, S]], attention alignments.
                durations: [torch.float32; [B, S]], durations.
                factor: [torch.float32; [B]], size ratio between ground-truth and predicted lengths.
        """
        # sample speaker embedding
        # [B, E]
        if spkid is not None:
            spkembed = self.spkembed(spkid)
        elif modalid is not None:
            spkembed = self.prior_mean(modalid)
        else:
            spkembed = self.prior_sample(text.size(0))
        return self.nat(text, textlen, spkembed, mel, mellen)

    def spkprior(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Speaker prior.
        Returns:
            weight: [torch.float32; [K]], weights of each modals.
            mean: [torch.float32; [K, E]], mean vectors.
            std: [torch.float32; [K, E]], standard deviations.
        """
        # [K]
        weight = torch.softmax(self.priorbuffer[:, 0], dim=0)
        # [K, E], [K, E]
        mean, logstd = self.priorbuffer[:, 1:].chunk(2, dim=-1)
        # [K, E]
        std = F.softplus(logstd)
        # [K], [K, E], [K, E]
        return weight, mean, std

    def prior_mean(self, modals: torch.Tensor) -> torch.Tensor:
        """Sample from prior mean.
        Args:
            modals: [torch.long; [B]], modal indices.
        Returns:
            [torch.float32; [B, E]], mean sampled.
        """
        _, mean, _ = self.spkprior()
        return mean[modals]

    def prior_sample(self,
                     batch: Optional[int] = None,
                     noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from prior.
        Args:
            batch: batch size of the sample.
            noise: [torch.float32; [B, K, E]], reparametrization basis,
                if both batch and noise are not None, return sample based on noise.
        Returns:
            [torch.float32; [B, E]], sampled prior.
        Exceptions:
            AssertionError, both batch and noise are None.
        """
        assert batch is None and noise is None, \
            'both batch and noise are None'
        # [K], [K, E], [K, E]
        weight, mean, std = self.spkprior()
        # K, E
        modal, embed = mean.shape
        if noise is None:
            # [B, K, E]
            noise = torch.zeros(batch, modal, embed, device=self.weight.device)
        # [B, K, E]
        reparam = mean[None] + std[None] * noise
        # [B, E]
        return (reparam * weight[None, :, None]).sum(dim=1)
