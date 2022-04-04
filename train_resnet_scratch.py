from typing import Optional

import numpy as np
import torchvision.models
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import torch
import argparse
import json
import multiprocessing

from torch.nn import functional as F

import sys

sys.path.append('pycil')
from pycil.trainer import _set_random, setup_train_device, print_args
from pycil.utils.data_manager import DataManager


class Model(LightningModule):

    def __init__(self, args, model=None, lr=1e-3, gamma=0.7, batch_size=128):
        super().__init__()
        # Save args
        self.args = args
        # Datamanager
        self.data_manager = None
        # Define network backbone
        self.save_hyperparameters(ignore="model")
        self.model_ft = torchvision.models.resnet18(pretrained=True)
        self.model_ft.fc = torch.nn.Linear(
            in_features=self.model_ft.fc.in_features,
            out_features=args['init_cls']
        )
        assert list(self.model_ft.children())[-1].out_features == args['init_cls']

    def forward(self, x):
        return self.model_ft(x)

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        # Accuracy
        _, preds = torch.max(logits, dim=1)
        n_correct = preds.eq(y.expand_as(preds)).cpu().sum()
        train_acc = n_correct / y.size(dim=0)
        self.log("train_loss", loss)
        self.log("train acc", train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        # Accuracy
        _, preds = torch.max(logits, dim=1)
        n_correct = preds.eq(y.expand_as(preds)).cpu().sum()
        val_acc = n_correct / y.size(dim=0)
        self.log("val_loss", loss)
        self.log("train acc", val_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def setup(self, stage: Optional[str] = None) -> None:
        self.data_manager = DataManager(
            self.args['dataset'], self.args['shuffle'], self.args['seed'], self.args['init_cls'], self.args['increment']
        )

    def _init_dataloader(self, split):
        return DataLoader(
            self.data_manager.get_dataset(indices=np.arange(0, self.args['init_cls']), source=split, mode=split),
            batch_size=self.hparams.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataloader = self._init_dataloader('train')
        assert np.unique(train_dataloader.dataset.labels).size == self.args['init_cls']
        return train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        test_dataloader = self._init_dataloader('test')
        assert np.unique(test_dataloader.dataset.labels).size == self.args['init_cls']
        return test_dataloader


def train(args):
    # Set up training
    args['device'] = setup_train_device(args)
    _set_random()
    # Print args
    print_args(args)
    # Model
    model = Model(args)
    # Training
    wandb_logger = WandbLogger()
    trainer = Trainer(log_every_n_steps=1, logger=wandb_logger)
    trainer.fit(model)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='pycil/exps/CIL_LogoDet-3k.json',
                        help='Json file of settings.')
    return parser


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


if __name__ == '__main__':
    main()
