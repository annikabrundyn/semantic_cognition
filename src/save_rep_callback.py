from pytorch_lightning.callbacks import Callback

class MyPrintingCallback(Callback):

    def on_epoch_end(self, trainer, pl_module):
        ### TODO
        pass