from abc import ABC
from functools import cached_property

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW

from utils.metrics import NIRRelatedNodeMetric


class BaseModel(pl.LightningModule, ABC):
    def __init__(self, **_):
        super().__init__()
        self.val_nir_metric = self.METRIC_CLASS(prefix="val_")
        self.test_nir_metric = self.METRIC_CLASS(prefix="test_")

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)

    def validation_epoch_end(self, *args, **kwargs) -> None:
        if self.global_step != 0:
            self.log_dict(
                self.val_nir_metric,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                batch_size=self.trainer.datamodule.val_batch_size,
                sync_dist=True,
            )
        self.val_nir_metric.reset()
        return super().validation_epoch_end(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs) -> None:
        self.log_dict(
            self.test_nir_metric,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=self.trainer.datamodule.val_batch_size,
            sync_dist=True,
        )
        self.test_nir_metric.reset()
        return super().test_epoch_end(*args, **kwargs)

    @cached_property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs


class SEENBaseModel(BaseModel, ABC):
    METRIC_CLASS = NIRRelatedNodeMetric

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5, device=self.device))
        self.ce_fn = nn.CrossEntropyLoss(torch.tensor([0.84, 45.12, 0.56], device=self.device))
        self.related_node_ce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(24.63))
        self.loss_weight1 = 0.5
        self.loss_weight2 = 0.5

    def calculate_loss(self, pre_retold_logit, post_retold_logit, batch):
        if batch["pre_retold_idx"] and batch["post_retold_idx"]:
            pre_retold_loss = self.bce_fn(
                pre_retold_logit[batch["pre_retold_idx"]].reshape(-1),
                batch["NIR_label"][batch["pre_retold_idx"]].float(),
            )
            post_retold_loss = self.ce_fn(
                post_retold_logit[batch["post_retold_idx"]],
                batch["NIR_label"][batch["post_retold_idx"]],
            )
            loss = (
                pre_retold_loss * len(batch["pre_retold_idx"]) / self.trainer.datamodule.batch_size
                + post_retold_loss
                * len(batch["post_retold_idx"])
                / self.trainer.datamodule.batch_size
            )
        elif batch["pre_retold_idx"]:
            loss = self.bce_fn(pre_retold_logit.reshape(-1), batch["NIR_label"].float())
        elif batch["post_retold_idx"]:
            loss = self.ce_fn(post_retold_logit, batch["NIR_label"])
        else:
            assert False
        return loss

    def calculate_related_node_loss(self, related_node_logits, related_nodes):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        pre_retold_logit, post_retold_logit, related_node_logits = self(**batch)
        nir_loss = self.calculate_loss(pre_retold_logit, post_retold_logit, batch)
        self.log(
            "nir_loss",
            nir_loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        # Related node loss
        related_node_logits = torch.tensor_split(
            related_node_logits, batch["graph_split_idx"].cpu()
        )
        related_node_loss = self.calculate_related_loss(related_node_logits, batch["related_nodes"])
        self.log(
            "related_node_loss",
            related_node_loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        loss = self.loss_weight1 * nir_loss
        if related_node_loss != 0:
            loss += self.loss_weight2 * related_node_loss

        self.log(
            "loss", loss, on_step=True, on_epoch=True, batch_size=self.trainer.datamodule.batch_size
        )
        return loss

    def evaluate(self, batch):
        with torch.no_grad():
            pre_retold_logit, post_retold_logit, related_node_logits = self(**batch)
        pred = [
            (pre_retold_logit[idx] > 0.5).int().item()
            if story_type == "pre-retold"
            else post_retold_logit[idx].argmax().item()
            for idx, story_type in enumerate(batch["story_type"])
        ]
        related_node_pred = torch.tensor_split(
            related_node_logits.reshape(-1) > 0.5, batch["graph_split_idx"].cpu()
        )
        return pred, related_node_pred

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred, related_node_pred = self.evaluate(batch)
        self.val_nir_metric.update(
            batch["pair_id"],
            batch["story_type"],
            batch["event_id"],
            [i.shape[0] for i in batch["related_nodes"]],
            batch["NIR_label"].tolist(),
            pred,
            [i.detach().cpu().tolist() for i in batch["related_nodes"]],
            [i.detach().cpu().tolist() for i in related_node_pred],
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pred, related_node_pred = self.evaluate(batch)
        self.test_nir_metric.update(
            batch["pair_id"],
            batch["story_type"],
            batch["event_id"],
            [i.shape[0] for i in batch["related_nodes"]],
            batch["NIR_label"].tolist(),
            pred,
            [i.detach().cpu().tolist() for i in batch["related_nodes"]],
            [i.detach().cpu().tolist() for i in related_node_pred],
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred, related_node_pred = self.evaluate(batch)
        return [
            {
                "pair_id": batch["pair_id"][idx],
                "story_type": batch["story_type"][idx],
                "event_id": batch["event_id"][idx],
                "label": batch["NIR_label"][idx].item(),
                "pred": pred,
                "num_nodes": [i.shape[0] for i in batch["related_nodes"]],
                "related_node": batch["related_nodes"][idx].detach().cpu().tolist(),
                "related_node_pred": related_node_pred[idx].detach().cpu().tolist(),
            }
            for idx, pred in enumerate(pred)
        ]
