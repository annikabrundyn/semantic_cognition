import os
from collections import defaultdict

import numpy as np
import torch

from pytorch_lightning.callbacks import Callback


class SaveRepCallback(Callback):

    def on_train_epoch_end(self, trainer, pl_module, outputs):

        # at end, reset and save results
        if (trainer.current_epoch + 1) % pl_module.hparams.save_epoch_freq == 0:

            base_dir = os.path.split(trainer.checkpoint_callback.dirpath)[0]
            epoch_path = os.path.join(base_dir, f"epoch_{trainer.current_epoch + 1}")
            os.makedirs(epoch_path)

            d = pl_module.store_avg_reps.items().cpu()

            for item_key, rep in d:

                # avg the values
                rep = torch.div(rep, pl_module.hparams.imgs_per_item)
                rep = rep.numpy()
                np.save(f"{epoch_path}/{item_key}", rep)

            print(pl_module.count)
            pl_module.count = 0
            pl_module.store_avg_reps = defaultdict(lambda: torch.zeros((pl_module.net.rep3d_shape),
                                                                       requires_grad=False,
                                                                       device=pl_module.device))




