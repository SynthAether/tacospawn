import torch
import torch.nn as nn


class Cbhg(nn.Module):
    """Convolutional bank, highway and bidirectional GRU.
    """
    def __init__(self,
                 channels: int,
                 banks: int,
                 pool: int,
                 kernels: int,
                 highways: int):
        """Initializer.
        Args:
            channels: size of the hidden channels.
            banks: size of the convolutional banks.
            pool: size of the maxpool widths.
            kernels: size of the convolutional kernels for conv projection.
            highways: the number of the highway layers.
        """
        super().__init__()
        self.banks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, k, padding=k // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU())
            for k in range(1, banks + 1)])
        self.proj = nn.Sequential(
            nn.MaxPool1d(pool, stride=1, padding=pool // 2),
            nn.Conv1d(banks * channels, channels, kernels, padding=kernels // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernels, padding=kernels // 2),
            nn.BatchNorm1d(channels))

        self.highway = nn.ModuleList([
            nn.Conv1d(channels, channels * 2, 1)
            for _ in range(highways)])
        self.bigru = nn.GRU(channels, channels, batch_first=True, bidirectional=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pass the inputs to convolutional-bank, highway and BiGRU.
        Args:
            inputs: [torch.float32; [B, T, C]], input tensor.
        Returns:
            [torch.float32; [B, T, C x 2]], transformed.
        """
        # T
        timestep = inputs.size(1)
        # [B, C, T]
        x = inputs.transpose(1, 2)
        # [B, K x C, T]
        bank = torch.cat([kgram(x)[..., :timestep] for kgram in self.banks], dim=1)
        # [B, C, T]
        x = x + self.proj(bank)[..., :timestep]
        # [B, C, T]
        for proj in self.highway:
            # [B, C, T], [B, C, T]
            context, logit = proj(x).chunk(2, dim=1)
            # [B, C, T]
            gate = torch.sigmoid(logit)
            # [B, C, T]
            x = context * gate + x * (1. - gate)
        # [B, T, C x 2]
        outputs, _ = self.bigru(x.transpose(1, 2))
        return outputs
