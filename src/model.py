from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from src.datamodule import SemanticDataModule
from src.model_components import Net


class BaseModel(pl.LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.net = Net(feat_extractor='resnet', img_size=64, hidden_size=128)
        self.criterion = nn.MSELoss()

    def forward(self, img, rel):
        return self.net(img, rel)

    def training_step(self, batch, batch_idx):

        # call forward
        pred, hidden, rep = self(batch['img'], batch['rel'])

        loss = self.criterion(pred, batch['attr'])
        return loss

    # def validation_step(self, *args, **kwargs):
    #     pass

    # def test_step(self, *args, **kwargs):
    #     pass

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [opt]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--root_dir", type=str, help='path to data folder', default='data')
        parser.add_argument("--feat_extractor", type=str, default='simple', choices=['simple', 'resnet18'])
        parser.add_argument("--crop_size", type=int, help='size of cropped square input images', default=64)
        parser.add_argument("--imgs_per_item", type=int, help='number of examples per item category', default=5)

        # hyperparameters
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--num_workers", type=int, default=0)


if __name__ == "__main__":

    parser = ArgumentParser()

    dm = SemanticDataModule('../data', imgs_per_item=5, batch_size=2, num_workers=0)
    dm.prepare_data()

    model = BaseModel()

    trainer = pl.Trainer()
    trainer.fit(model, dm.train_dataloader())