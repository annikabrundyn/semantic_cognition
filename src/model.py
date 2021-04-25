from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch import nn
import pytorch_lightning as pl

from datamodule import SemanticDataModule
from model_components import Net
from save_rep_callback import SaveRepCallback


class BaseModel(pl.LightningModule):
    def __init__(self,
                 feat_extractor: str,
                 imgs_per_item: int,
                 crop_size: int,
                 hidden_size: int,
                 lr: float,
                 save_epoch_freq: int,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.net = Net(feat_extractor=self.hparams.feat_extractor,
                       crop_size=self.hparams.crop_size,
                       hidden_size=self.hparams.hidden_size)

        # TODO: not sure what the loss should be - in hw nn.MSELoss
        self.criterion = nn.MultiLabelSoftMarginLoss()

        self.store_avg_reps = defaultdict(lambda: torch.zeros((32, 29, 29), requires_grad=False))
        self.count = 0

    def forward(self, img, rel):
        return self.net(img, rel)

    def training_step(self, batch, batch_idx):

        # call forward
        pred, hidden, rep = self(batch['img'], batch['rel'])

        loss = self.criterion(pred, batch['attr'])

        # save representations
        if (self.trainer.current_epoch + 1) % self.hparams.save_epoch_freq == 0:
            for i, item in enumerate(batch['item_name']):
                self.store_avg_reps[item] += rep[i].detach()
                self.count += 1

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

        parser.add_argument("--root_dir", type=str, help='path to data folder', default='../data')
        parser.add_argument("--feat_extractor", type=str, default='simple', choices=['simple', 'resnet'])
        parser.add_argument("--crop_size", type=int, help='size of cropped square input images', default=64)
        parser.add_argument("--hidden_size", type=int, help='size of cropped square input images', default=128)
        parser.add_argument("--imgs_per_item", type=int, help='number of examples per item category', default=20)
        parser.add_argument("--save_epoch_freq", type=int, help='how often to save representations', default=50)

        # hyperparameters
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--seed", type=int, default=98264)

        return parser



if __name__ == "__main__":

    # parse args
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser = BaseModel.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # data
    dm = SemanticDataModule(**args.__dict__)
    dm.prepare_data()

    # model
    model = BaseModel(**args.__dict__)

    # train
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[SaveRepCallback()])
    trainer.fit(model, dm.train_dataloader())