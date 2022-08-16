import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    def on_train_start(self, *args, **kwargs) -> None:
        pass

    def on_train_batch_end(self, *args, **kwargs) -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.global_step != 0:
            monitor_candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, monitor_candidates)
