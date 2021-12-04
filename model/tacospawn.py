import torch
import torch.nn as nn

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

