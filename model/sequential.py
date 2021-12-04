from typing import List

import torch.nn as nn


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
