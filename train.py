import argparse
import json
import os

import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from tacospawn import TacoSpawn
from utils.libritts import LibriTTS, LibriTTSDataset
from utils.wrapper import TrainingWrapper


class Trainer:
    """TacoSpawn trainer.
    """
    def __init__(self,
                 model: TacoSpawn,
                 dataset: LibriTTSDataset,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: tacospawn model.
            dataset: multispeaker dataset.
            config: unified configurations.
            device: target computing device.
        """
        self.model = model
        self.dataset = dataset
        self.config = config

        self.wrapper = TrainingWrapper(model, device)

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.train.batch,
            shuffle=config.train.shuffle,
            collate_fn=self.dataset.collate,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            config.train.learning_rate,
            (config.train.beta1, config.train.beta2),
            config.train.eps)

        self.train_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def train(self, epoch: int = 0):
        """Train wavegrad.
        Args:
            epoch: starting step.
        """
        self.model.train()
        step = epoch * len(self.loader)
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=len(self.loader), leave=False) as pbar:
                for it, bunch in enumerate(self.loader):
                    loss, losses, aux = self.wrapper.compute_loss(bunch)
                    # update
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    step += 1
                    pbar.update()
                    pbar.set_postfix({'loss': loss.item(), 'step': step})

                    for name, loss in losses.items():
                        self.train_log.add_scalar(f'loss/{name}', loss.item(), step)
                    
                    with torch.no_grad():
                        grad_norm = np.mean([
                            torch.norm(p.grad).item()
                            for p in self.model.parameters() if p.grad is not None])
                        param_norm = np.mean([
                            torch.norm(p).item()
                            for p in self.model.parameters() if p.dtype == torch.float32])

                    self.train_log.add_scalar('common/grad-norm', grad_norm, step)
                    self.train_log.add_scalar('common/param-norm', param_norm, step)

                    if (it + 1) % (len(self.loader) // 100) == 0:
                        # [T, M]
                        mel = aux['mel'][0].cpu().detach().numpy()
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/mel', self.mel_img(mel).transpose(2, 0, 1), step)
                        # [T, S]
                        align = aux['align'][0].cpu().detach().numpy()
                        # [3, S, T]
                        align = self.cmap[(align * 255).astype(np.long)].transpose(2, 1, 0)
                        self.train_log.add_image('train/align', align, step)

                    if (it + 1) % (len(self.loader) // 10) == 0:
                        # wrapping
                        sid, text, _, textlen, _ = self.wrapper.wrap(bunch)
                        with torch.no_grad():
                            self.model.eval()
                            # [1, T, M]
                            pred, _, aux = self.model(
                                text[:1], textlen[:1], sid=sid[:1], sample=False)
                            self.model.train()
                        # [T, M]
                        pred = pred.cpu().detach().numpy().squeeze(0)
                        self.train_log.add_image(
                            # [3, M, T]
                            'eval/mel', self.mel_img(pred).transpose(2, 0, 1), step)
                        # [T, S]
                        align = aux['align'].cpu().detach().numpy().squeeze(0)
                        # [3, S, T]
                        align = self.cmap[(align * 255).astype(np.long)].transpose(2, 1, 0)
                        self.train_log.add_image('eval/align', align, step)
                        del pred

            self.model.save(
                '{}_{}.ckpt'.format(self.ckpt_path, epoch), self.optim)

    def mel_img(self, mel: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram images.
        Args:
            signal: [float32; [T, M]], speech signal.
        Returns:
            [float32; [M, T, 3]], mel-spectrogram in viridis color map.
        """
        # [M, T]
        mel = mel.transpose(1, 0)
        # minmax norm in range(0, 1)
        mel = (mel - mel.min()) / (mel.max() - mel.min())
        # in range(0, 255)
        mel = (mel * 255).astype(np.long)
        # [M, T // H, 3]
        mel = self.cmap[mel]
        # make origin lower
        mel = np.flip(mel, axis=0)
        return mel


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-epoch', default=0, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--auto-rename', default=False, action='store_true')
    args = parser.parse_args()

    # seed setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # configurations
    config = Config(LibriTTS.count_speakers(args.data_dir))
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    if args.name is not None:
        config.train.name = args.name

    log_path = os.path.join(config.train.log, config.train.name)
    # auto renaming
    if args.auto_rename and os.path.exists(log_path):
        config.train.name = next(
            f'{config.train.name}_{i}' for i in range(1024)
            if not os.path.exists(f'{log_path}_{i}'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # prepare datasets
    libritts = LibriTTSDataset(args.data_dir, config.data)

    # model definition
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tacospawn = TacoSpawn(config.model)
    tacospawn.to(device)

    trainer = Trainer(tacospawn, libritts, config, device)

    # loading
    if args.load_epoch > 0:
        # find checkpoint
        ckpt_path = os.path.join(
            config.train.ckpt,
            config.train.name,
            f'{config.train.name}_{args.load_epoch}.ckpt')
        # load checkpoint
        ckpt = torch.load(ckpt_path)
        tacospawn.load(ckpt, trainer.optim)
        print('[*] load checkpoint: ' + ckpt_path)
        # since epoch starts with 0
        args.load_epoch += 1

    # git configuration
    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    # start train
    trainer.train(args.load_epoch)
