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
        # alias
        self.reduction = model.taco.reduction.factor

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
        ## wrapping
        sid, text, mel, textlen, mellen = self.wrap(bunch)
        # for alignment endpoint loss
        mel = F.pad(mel, [0, 0, 0, self.reduction])
        # B, T, _
        bsize, timestep, _ = mel.shape
        # [B, T]
        mel_mask = (
            torch.arange(timestep, device=mel.device)[None]
            < mellen[:, None].to(torch.float32))
        # mask with silence
        # => additional padding for spectrogram reduction should use silence pad value
        mel.masked_fill_(~mel_mask[..., None].to(torch.bool), np.log(1e-5))

        ## outputs
        masked_mel, _, aux = self.model(text, textlen, mel, mellen, sid=sid, sample=True)
        # 1. mel spectrogram loss
        rctor = F.l1_loss(mel, aux['unmasked'])
        # 2. prior matching
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
        # 3. entropy loss
        mean, std = aux['speaker']['mean'], aux['speaker']['std']
        entropy = 2 * torch.log(std + 1e-5) + torch.square((sample - mean) / (std + 1e-5))
        entropy = entropy.mean()

        ## aggregation
        loss = rctor - likelihood - entropy
        losses = {
            'rctor': rctor.cpu().detach().numpy(),
            'likelihood': likelihood.cpu().detach().numpy(),
            'entropy': entropy.cpu().detach().numpy()}
        aux = {'mel': masked_mel, 'align': aux['align']}
        return loss, losses, aux 
