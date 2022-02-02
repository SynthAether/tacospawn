import argparse
import multiprocessing as mp
import os
from typing import Any, Callable, List, Tuple

import numpy as np
from tqdm import tqdm

from speechset import Config
from speechset.datasets import DataReader

from .dataset import LibriTTSDataset


class DumpReader(DataReader):
    """Dumped dataset reader.
    """
    def __init__(self, data_dir: str):
        """Initializer.
        Args:
            data_dir: dumped datasets.
        """
        self.filelists = [
            os.path.join(data_dir, path)
            for path in os.listdir(data_dir)
            if path.endswith('.npy')]

    def dataset(self) -> List[str]:
        """Return file reader.
        Returns:
            file-format datum read.er
        """
        return self.filelists
    
    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns;
            preprocessor, load npy and return normalized.
        """
        return self.preprocessor
    
    def preprocessor(self, path: str) -> Tuple[int, np.ndarray, np.ndarray]:
        """Load dumped datum.
        Args:
            path: path to the npy file.
        Returns:
            tuple,
                sid: speaker id.
                labels: [np.int32, [S]], labled texts.
                mel: [np.float32; [T, M]], mel-spectrogram.
        """
        dump = np.load(path, allow_pickle=True).item()
        return dump['sid'], dump['labels'], dump['mel']


class DumpDataset(LibriTTSDataset):
    """Dumped dataset.
    """
    def __init__(self, data_dir: str, config: Config):
        """Initializer.
        Args:
            data_dir: path to the dumped dataset.
            config: configurations.
        """
        super(LibriTTSDataset, self).__init__(DumpReader(data_dir), config)
    
    def normalize(self, *args) -> List[Any]:
        """Identity map.
        """
        return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-proc', default=4, type=int)
    args = parser.parse_args()

    config = Config(batch=None)
    libritts = LibriTTSDataset(args.data_dir, config)

    os.makedirs(args.output_dir, exist_ok=True)

    def dumper(i):
        sid, labels, mel = libritts[i]
        np.save(
            os.path.join(args.output_dir, f'{i}.npy'),
            {'sid': sid, 'labels': labels, 'mel': mel})
        return i

    length = len(libritts)
    with mp.Pool(args.num_proc) as pool:
        for _ in tqdm(pool.imap_unordered(dumper, range(length)), total=length):
            pass
