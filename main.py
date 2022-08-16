import json
import os
from itertools import chain
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.multiprocessing
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from transformers.hf_argparser import HfArgumentParser

from modeling.base import BaseModel
from settings import Arguments
from utils.callbacks import Checkpoint

torch.multiprocessing.set_sharing_strategy("file_system")


def main(args: Arguments):
    model_class: pl.LightningModule = args.model_class
    dm: pl.LightningDataModule = args.datamodule

    checkpoint_dir = f"{args.tb_logger.log_dir}/checkpoints"
    best_model_path = Path(args.tb_logger.log_dir).absolute().parent / "best.ckpt"

    # init model
    if args.do_train:
        model: BaseModel = model_class(pretrained_path=args.experiment.pretrained_path)
        main_metric = f"val_{model.METRIC_CLASS.MAIN_EVAL_METRIC}"
        callback = Checkpoint(
            dirpath=checkpoint_dir,
            monitor=main_metric,
            filename=f"{{epoch}}-{{step}}-{{{main_metric}:.5f}}",
            mode="max",
            save_top_k=3,
            save_weights_only=True,
            auto_insert_metric_name=True,
        )
    else:
        model_path = args.test_model_path if args.test_model_path else best_model_path
        model: BaseModel = model_class.load_from_checkpoint(
            model_path,
        )
        callback = None

    trainer = pl.Trainer(
        logger=args.loggers,
        gpus=args.gpus if args.do_train else min(1, args.gpus),
        num_nodes=1,
        precision=16,
        strategy=DDPStrategy(),
        min_epochs=1,
        max_epochs=args.epochs,
        callbacks=callback,
        accumulate_grad_batches=args.accumulate_grad_batch,
        val_check_interval=args.val_step,
        fast_dev_run=8 if args.dev else False,
    )
    if trainer.is_global_zero:
        print(args)

    if args.do_train:
        trainer.fit(model=model, datamodule=dm)
        if trainer.is_global_zero:
            print(callback.best_model_path)
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            os.symlink(callback.best_model_path, best_model_path)
    else:
        if args.do_val:
            dm.setup(TrainerFn.VALIDATING)
            predictions = trainer.predict(model=model, dataloaders=dm.val_dataloader())
            metric_prefix = "val_"
        elif args.do_test:
            dm.setup(TrainerFn.TESTING)
            predictions = trainer.predict(model=model, dataloaders=dm.test_dataloader())
            metric_prefix = "test_"
        else:
            raise ValueError("Invalid job type")

        predictions = list(chain.from_iterable(predictions))
        metric_class = model.METRIC_CLASS
        df = metric_class.get_metric_df(predictions)
        metric = metric_class.get_metric(df, prefix=metric_prefix)
        if args.test_model_path:
            prefix = args.test_model_path.rsplit(".", maxsplit=1)[0]
        else:
            prefix = os.readlink(best_model_path).rsplit(".", maxsplit=1)[0]
        if args.dev:
            return
        df.to_csv(f"{prefix}_{args.job_type}_pred.csv", index=False, doublequote=True)
        with open(f"{prefix}_{args.job_type}_metric.json", "w") as f:
            json.dump(metric, f)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    main(args)
