from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from pytorch_lightning.trainer.states import TrainerFn
from torch._six import string_classes
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx, to_networkx

from datamodule.dataset import DictDataset
from datamodule.loader import Collater

torch.multiprocessing.set_sharing_strategy("file_system")
MAX_LEN = 1536
MAX_TOKENS_PER_NODE = 96


def stack_ndarray(data):
    if isinstance(data, Mapping):
        return {k: stack_ndarray(data[k]) for k in data}
    elif isinstance(data, Sequence) and not isinstance(data, string_classes):
        return [stack_ndarray(i) for i in data]
    elif isinstance(data, np.ndarray) and data.dtype == np.object0:
        if all(len(data[0]) == len(i) for i in data):
            return np.vstack(data)
        else:
            return [np.vstack(i) for i in data]
    return data


def read_parquet(file_path, columns=None) -> pd.DataFrame:
    return (
        pyarrow.parquet.read_table(file_path, columns=columns).to_pandas().applymap(stack_ndarray)
    )


_collate_fn = Collater()
BASE_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = BASE_DIR / "data"


class BaseDataModule(pl.LightningDataModule):
    LOADER_KWARGS = dict(num_workers=5, persistent_workers=True, pin_memory=True)
    DATA_DIR = DATA_DIR

    def __init__(self, batch_size=5, val_batch_size=5):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

    def collate_fn(self, batch):
        return _collate_fn(batch)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            **self.LOADER_KWARGS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.LOADER_KWARGS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.LOADER_KWARGS,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.LOADER_KWARGS,
        )


class NIRDataModule(BaseDataModule):
    TENSOR_FIELDS = {"input_ids", "mask", "global_mask", "NIR_label"}
    LIST_TENSOR_FIELDS = {"article_node2token", "article_node2token_mask", "related_nodes"}
    LIST_FIELDS = {"pair_id", "story_type", "event_id"}
    article_columns = [
        "pair_id",
        "story_type",
        "edge_index",
        "input_ids",
        "mask",
        "node2token",
        "graph",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        self.articles = {
            pair_id: {
                story_type: article.to_dict()
                for story_type, article in articles.set_index("story_type").iterrows()
            }
            for pair_id, articles in read_parquet(
                DATA_DIR / "article.parquet", columns=self.article_columns
            ).groupby("pair_id")
        }

        if stage == TrainerFn.FITTING:
            self.train_dataset = DictDataset(
                read_parquet(DATA_DIR / "train.parquet").to_dict("records")
            )
            self.val_dataset = DictDataset(
                read_parquet(DATA_DIR / "val.parquet").to_dict("records")
            )
        elif stage == TrainerFn.VALIDATING:
            self.val_dataset = DictDataset(
                read_parquet(DATA_DIR / "val.parquet").to_dict("records")
            )
        elif stage == TrainerFn.TESTING or stage == TrainerFn.PREDICTING:
            self.test_dataset = DictDataset(
                read_parquet(DATA_DIR / "test.parquet").to_dict("records")
            )

    def prepare_input(self, data):
        invert_story_type = "pre-retold" if data["story_type"] == "post-retold" else "post-retold"
        article = self.articles[data["pair_id"]][invert_story_type]
        event_input_id_len = len(data["input_ids"])
        data["input_ids"] = np.concatenate((data["input_ids"], article["input_ids"][1:]))[:MAX_LEN]
        data["mask"] = np.concatenate((data["mask"], article["mask"][1:]))[:MAX_LEN]
        data["global_mask"] = np.zeros(MAX_LEN, dtype=np.int16)
        data["global_mask"][:event_input_id_len] = 1
        data["article_node2token"] = np.where(
            article["node2token"] == -1,
            article["node2token"],
            article["node2token"] + event_input_id_len - 1,
        )
        data["article_node2token_mask"] = np.where(
            article["node2token"] == -1,
            np.ones_like(article["node2token"], dtype=np.int8),
            np.zeros_like(article["node2token"], dtype=np.int8),
        )

        assert_idx = np.random.choice(len(data["article_node2token"]), 1)[0]
        assert_range = np.random.choice(MAX_TOKENS_PER_NODE, 1)[0]
        assert (
            data["input_ids"][data["article_node2token"][assert_idx][:assert_range]]
            == article["input_ids"][article["node2token"][assert_idx][:assert_range]]
        ).all()

        # insert super node
        data["article_node2token"] = np.vstack(
            [np.full((1, MAX_TOKENS_PER_NODE), -1), data["article_node2token"]]
        )
        data["article_node2token_mask"] = np.where(
            data["article_node2token"] == -1,
            np.ones_like(data["article_node2token"], dtype=np.int8),
            np.zeros_like(data["article_node2token"], dtype=np.int8),
        )
        graph = Data(edge_index=torch.tensor(article["edge_index"], dtype=torch.long))
        graph.num_nodes = len(article["graph"])
        raw_adj = nx.to_numpy_array(to_networkx(graph))
        adj = np.zeros((raw_adj.shape[0] + 1, raw_adj.shape[0] + 1), dtype=np.int8)
        adj[0] = adj[:, 0] = 1
        adj[1:, 1:] = raw_adj
        data["graph"] = from_networkx(nx.from_numpy_array(adj))
        data["graph"].num_nodes = len(adj)
        data["related_nodes"] = np.concatenate(
            [[0], data["related_nodes"][: data["graph"].num_nodes - 1]]
        )
        assert (data["related_nodes"] >= 0).all()
        return data

    def collate_fn(self, batch_):
        batch = []
        graph_split_idx = [0]
        for data in deepcopy(batch_):
            data = self.prepare_input(data)
            graph_split_idx.append(data["graph"].num_nodes + graph_split_idx[-1])
            batch.append(data)

        return {
            "graph": Batch.from_data_list([i["graph"] for i in batch]),
            "graph_split_idx": torch.tensor(graph_split_idx[1:-1], dtype=torch.long),
            "super_node_idx": torch.tensor(graph_split_idx[:-1], dtype=torch.long),
            "pre_retold_idx": [
                idx for idx, i in enumerate(batch) if i["story_type"] == "pre-retold"
            ],
            "post_retold_idx": [
                idx for idx, i in enumerate(batch) if i["story_type"] == "post-retold"
            ],
            **{
                attr: torch.tensor(np.array([i[attr] for i in batch], dtype=np.int_))
                for attr in self.TENSOR_FIELDS
            },
            **{
                attr: [torch.tensor(i[attr].astype(dtype=np.int_)) for i in batch]
                for attr in self.LIST_TENSOR_FIELDS
            },
            **{attr: [i[attr] for i in batch] for attr in self.LIST_FIELDS},
        }
