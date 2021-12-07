from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tacospawn import TacoSpawn


class TrainingWrapper:
    """TacoSpawn training wrapper.
    """
    def __init__(self, model: TacoSpawn, device: torch.device):
        """Initializer.
        Args:
            model: tacospawn model.
            device: torch device.
        """
        self.model = model
        self.device = device

    def wrap(self, bunch: List[np.ndarray]) -> List[torch.Tensor]:
        """Wrap the array to torch tensor.
        Args:
            bunch: input tensors.
        Returns:
            wrapped.
        """
        return [torch.tensor(array, device=self.device) for array in bunch]

    def compute_loss(self, bunch: List[np.ndarray]) -> Tuple[torch.Tensor, Dict[str, np.float32]]:
        """Compute unconditional VLB-TacoSpawn loss.
        Args:
            bunch: input tensors.
                sid: [np.long; [B]], speaker id.
                text: [np.long; [B, S]], input text tokens.
                textlen: [np.long; [B]], lengths of each texts.
                mel: [np.float32; [B, T, M]], mel-spectrograms.
                mellen: [np.long; [B]], length of each mel-spectrograms.
        Returns:
            loss tensor and details.
        """
        # wrapping
        sid, text, mel, textlen, mellen = self.wrap(bunch)
        # outputs
        pred, predlen, aux = self.model(text, textlen, mel, mellen, sid=sid, sample=True)
        # 1. mel spectrogram loss
        rctor = F.l1_loss(mel, pred)
        # 2. factor loss, length matching loss
        factor = torch.square(aux['factor']).mean()
        # 3. prior matching
        # [B, E]
        sample = aux['speaker']['sample']
        # [K], [K, E], [K, E]
        weight, mean, std = self.model.parametrize(self.model.priorbuffer)
        # [B, K, E]
        ll = -2 * torch.log(std[None] + 1e-5) - \
            torch.square((sample[:, None] - mean[None]) / (std[None] + 1e-5))
        # [B, E]
        gmm = (weight[None, :, None] * torch.exp(ll)).sum(dim=1)
        likelihood = torch.log(gmm + 1e-5).mean()
        # 4. entropy loss
        mean, std = aux['speaker']['mean'], aux['speaker']['std']
        entropy = -2 * torch.log(std + 1e-5) - torch.square((sample - mean) / (std + 1e-5))
        entropy = entropy.mean()
        return rctor + factor + likelihood + entropy, {
            'rctor': rctor.cpu().detach().numpy(),
            'factor': factor.cpu().detach().numpy(),
            'likelihood': likelihood.cpu().detach().numpy(),
            'entropy': entropy.cpu().detach().numpy()}
