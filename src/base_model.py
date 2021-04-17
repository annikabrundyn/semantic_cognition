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
        self.net = Net(10, 10)
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



if __name__ == "__main__":

    parser = ArgumentParser()

    dm = SemanticDataModule('../data', imgs_per_item=5, batch_size=2, num_workers=0)
    dm.prepare_data()

    model = BaseModel()

    trainer = pl.Trainer()
    trainer.fit(model, dm.train_dataloader())