import torch
import torch.nn as nn

from .cbhg import Cbhg
from .config import Config
from .misc import Prenet
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

        self.upsampler = Upsampler(
            config.channels, config.upsampler_layers)

    def forward(self, inputs: torch.Tensor, textlen: torch.Tensor) -> torch.Tensor:
        """Encode text tokens.
        Args;
            inputs: [torch.long; [B, S]], text symbol sequences.
            textlen: [torch.long; [B]], sequence lengths.
        Returns:
            [torch.float32; [B, S, C x 2]], CBHG features.
        """
        # [B, S, E]
        embed = self.embedding(inputs)
        # [B, S, C]
        preproc = self.prenet(embed)
        # [B, S, C x 2]
        return self.cbhg(preproc)
