from typing import List, Tuple

import numpy as np

from speechset import AcousticDataset, Config

from .reader import LibriTTS


class LibriTTSDataset(AcousticDataset):
    """Dataset for text to acoustic features.
    """
    def __init__(self, data_dir: str, config: Config):
        """Initializer.
        Args:
            data_dir: path to the libritts datasets.
            config: configurations.
        """
        super().__init__(LibriTTS(data_dir), config)

    def normalize(self, sid: int, text: str, speech: np.ndarray) \
            -> Tuple[int, np.ndarray, np.ndarray]:
        """Normalize datum.
        Args:
            sid: speaker id.
            text: transcription.
            speech: [np.float32; [T]], speech in range (-1, 1).
        Returns:
            normalized datum.
                sid: speaker id.
                labels: [np.long; [S]], labeled text sequence.
                mel: [np.float32; [T // hop, mel]], mel spectrogram.
        """
        # [S], [T // hop, mel]
        labels, mel = super().normalize(text, speech)
        return sid, labels, mel

    def collate(self, bunch: List[Tuple[int, np.ndarray, np.ndarray]]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [...] list of normalized inputs.
                sid: speaker id.
                labels: [np.long; [Si]], labled text sequence.
                mel: [np.float32; [Ti, mel]], mel spectrogram.
        Returns:
            batch data.
                sid: [np.long; [B]], speaker ids.
                text: [np.long; [B, S]], labeled text sequence.
                mel: [np.float32; [B, T, mel]], mel spectrogram.
                textlen: [np.long; [B]], text lengths.
                mellen: [np.long; [B]], spectrogram lengths.
        """
        # [B]
        sid = np.array([sid for sid, _, _ in bunch], dtype=np.long)
        # [B], [B]
        textlen, mellen = np.array(
            [[len(labels), len(spec)] for _, labels, spec in bunch], dtype=np.long).T
        # [B, S]
        text = np.stack(
            [np.pad(labels, [0, textlen.max() - len(labels)]) for _, labels, _ in bunch])
        # [B, T, mel]
        mel = np.stack(
            [np.pad(spec, [[0, mellen.max() - len(spec)], [0, 0]]) for _, _, spec in bunch])
        return sid, mel, text, textlen, mellen
