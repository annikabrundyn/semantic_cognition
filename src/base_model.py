import torch
import pytorch_lightning as pl
import numpy as np


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