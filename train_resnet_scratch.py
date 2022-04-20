import logging
from typing import Optional

import numpy as np
import torchvision.models
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS

from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
import multiprocessing

from torch.nn import functional as F

import sys

from config import SEED

sys.path.append('pycil')
from pycil.trainer import _set_random, print_args, init_logger
from pycil.utils.data_manager import DataManager


experiment_args = {
    "run_name": "BASELINE-FAIR_resnet152-from_scratch-100_classes",
    "prefix": "reproduce",
    "dataset": "LogoDet-3K_cropped",
    "shuffle": True,
    "init_cls": 100,
    "increment": 10,
    "model_name": "der",
    "data_augmentation": True,
    "seed": SEED,

    # Grid search parameters
    "dropout": 0.5,
    "convnet_type": None,
    "pretrained": None,

    # Baseline method?
    "baseline": True
}


class Model(LightningModule):

    def __init__(self, args, model=None, lr=1e-3, gamma=0.7, batch_size=128):
        super().__init__()
        # Save args
        self.args = args
        self.save_hyperparameters(ignore="model")
        # Datamanager
        self.data_manager = None
        # Define network backbone
        self.resnet = torchvision.models.resnet152(pretrained=True)
        self.dropout = torch.nn.Dropout(self.args['dropout']) if self.args['dropout'] else None
        self.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features,
            out_features=self.args['init_cls']
        )
        assert self.fc.out_features == args['init_cls']

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        if self.dropout:
            x = self.dropout(x)

        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        # Accuracy
        _, preds = torch.max(logits, dim=1)
        n_correct = preds.eq(y.expand_as(preds)).cpu().sum()
        train_acc = n_correct / y.size(dim=0)
        return dict(loss=loss, train_acc=train_acc)

    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        # Accuracy
        _, preds = torch.max(logits, dim=1)
        n_correct = preds.eq(y.expand_as(preds)).cpu().sum()
        val_acc = n_correct / y.size(dim=0)
        return dict(loss=loss, val_acc=val_acc)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        last = outputs[-1]
        self.log("train_loss", last['loss'])
        self.log("train_acc", last['train_acc'])

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        last = outputs[-1]
        self.log("val_loss", last['loss'])
        self.log("val_acc", last['val_acc'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def setup(self, stage: Optional[str] = None) -> None:
        self.data_manager = DataManager(
            self.args['dataset'], self.args['shuffle'], self.args['seed'],
            self.args['init_cls'], self.args['increment'], data_augmentation=self.args['data_augmentation']
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
    # Init logger
    init_logger(args, 'pycil/logs')
    # Set up seed
    _set_random()
    # Print args
    print_args(args)
    # Model
    model = Model(args)
    logging.info('Network architecture')
    logging.info(model.resnet)
    logging.info(model.dropout)
    logging.info(model.fc)
    # Training
    trainer = Trainer(
        log_every_n_steps=1, accelerator='auto', max_epochs=150,
        logger=WandbLogger(
            project='pycil', name=args['run_name']),
        callbacks=[
            EarlyStopping(monitor="val_acc", min_delta=0.00, patience=30,
                          verbose=True, mode="max")
        ]
    )
    trainer.fit(model)


def main(args):
    # TODO: Multiple dropout
    train(args)


if __name__ == '__main__':
    main(experiment_args)
