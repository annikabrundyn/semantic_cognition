from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
import numpy as np

from src.datamodule import SemanticDataModule


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def forward(self):
        pass

    def training_step(self):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass


if __name__ == "__main__":

    parser = ArgumentParser()

    dm = SemanticDataModule('../data', imgs_per_item=5, batch_size=2, num_workers=0)
    dm.prepare_data()

    model = BaseModel()

    trainer = pl.Trainer()
    trainer.fit(model, dm.train_dataloader())