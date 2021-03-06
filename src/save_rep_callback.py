import os
from collections import defaultdict

import torch

from pytorch_lightning.callbacks import Callback


class SaveRepCallback(Callback):

    def __init__(self, dl_dict):
        self.dl_dict = dl_dict

    def on_train_epoch_end(self, trainer, pl_module, outputs):

        # at end, reset and save results
        if (trainer.current_epoch) % pl_module.hparams.save_epoch_freq == 0:
            print("generating representations and saving result")

            # create new dict for saving rep
            store_avg_reps = defaultdict(lambda: torch.zeros((pl_module.hparams.rep_size),
                                                             requires_grad=False,
                                                             device=pl_module.device))

            # create folder to store
            base_dir = os.path.split(trainer.checkpoint_callback.dirpath)[0]
            epoch_path = os.path.join(base_dir, f"epoch_{trainer.current_epoch}")
            os.makedirs(epoch_path)

            for item_name, dl in self.dl_dict.items():
                for batch in dl:

                    _, _, rep = pl_module(batch['img'].to(pl_module.device), batch['rel'].to(pl_module.device))

                    # element wise sum over batch
                    rep = torch.sum(rep, dim=0)
                    store_avg_reps[item_name] += rep

                store_avg_reps[item_name] = torch.div(store_avg_reps[item_name], len(dl.dataset))
                torch.save(rep, f"{epoch_path}/{item_name}.pt")
