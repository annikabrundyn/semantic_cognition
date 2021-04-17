import pytest
import pytorch_lightning as pl

from model import BaseModel
from datamodule import SemanticDataModule

@pytest.mark.parametrize("feat_extractor,crop_size,hidden_size,",
                         [('simple', 10, 10), ('resnet', 64, 10)])
def test_model(seed_everything, data_dir, feat_extractor, crop_size, hidden_size):

    dm = SemanticDataModule(data_dir,
                            imgs_per_item=5,
                            crop_size=crop_size,
                            seed=42,
                            batch_size=2,
                            num_workers=0)
    dm.prepare_data()

    model = BaseModel(feat_extractor, crop_size, hidden_size, lr=0.01)

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, dm)