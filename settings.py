import math
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Callable, Optional

import pytorch_lightning as pl
from pydantic import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule.event_graph import NIRDataModule
from modeling.seen import SEENLongformer, SEENLongformerLarge


class ExperimentSetting(BaseModel):
    datamodule: Callable
    model: Callable
    pretrained_path: Optional[str] = False

    class Config:
        arbitrary_types_allowed = True


PRETRAINED_MODEL_PATH = ""
PRETRAINED_LARGE_MODEL_PATH = ""

SEENLongformer = ExperimentSetting(datamodule=NIRDataModule, model=SEENLongformer)
SEENLongformerLarge = SEENLongformer.copy(update={"model": SEENLongformerLarge})
SEENLongformerPretrained = SEENLongformer.copy(update={"pretrained_path": PRETRAINED_MODEL_PATH})
SEENLongformerPretrained = SEENLongformerLarge.copy(
    update={"pretrained_path": PRETRAINED_LARGE_MODEL_PATH}
)
EXP_MAP = {
    "SEENLongformer": SEENLongformer,
    "SEENLongformerLarge": SEENLongformerLarge,
    "SEENLongformerPretrained": SEENLongformerPretrained,
    "SEENLongformerPretrained": SEENLongformerPretrained,
}


BASIC_BATCH = 8


@dataclass
class Arguments:
    # process
    do_train: bool = field(default=False)
    do_val: bool = field(default=False)
    do_test: bool = field(default=False)
    dev: bool = field(default=False)
    epochs: int = field(default=3)

    # device
    gpus: int = field(default=2)
    batch_size: int = field(default=2)
    val_batch_size: int = field(default=10)
    val_step: int = field(default=5)

    # experiment
    seed: int = field(default=301)
    group: Optional[str] = field(default=None)
    exp_name: str = field(default="SEENLongformer")
    test_model_path: str = field(default="")

    def __post_init__(self):
        pl.seed_everything(self.seed, workers=True)

        self.experiment: ExperimentSetting = EXP_MAP[self.exp_name]
        self.model_class = self.experiment.model
        self.datamodule = self.experiment.datamodule(self.batch_size, self.val_batch_size)

        self.accumulate_grad_batch = (
            None if self.batch_size > BASIC_BATCH else int(math.ceil(BASIC_BATCH / self.batch_size))
        )

        if sum([self.do_train, self.do_val, self.do_test]) == 0:
            self.do_train = self.dev = True
        assert sum([self.do_train, self.do_val, self.do_test]) == 1

        if self.do_train:
            self.job_type = "train"
        elif self.do_val:
            self.job_type = "val"
        elif self.do_test:
            self.job_type = "test"

    @cached_property
    def loggers(self):
        loggers = []
        if self.do_train:
            loggers = [self.tb_logger]
        return loggers

    @cached_property
    def tb_logger(self):
        logger = TensorBoardLogger(save_dir="tb_logs", name=self.exp_name)
        if self.do_train:
            logger.log_hyperparams(asdict(self))
        return logger
