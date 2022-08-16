import json
import sys
from itertools import chain
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet
import tokenizations
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from transformers import AutoTokenizer, LongformerTokenizer

model_name = "allenai/longformer-base-4096"
node_types = ["subject", "predicate", "object"]
tqdm.pandas()
BASE_DIR = Path().absolute().parent
DATA_DIR = BASE_DIR / "data"

if BASE_DIR.as_posix() not in sys.path:
    sys.path.append(BASE_DIR.as_posix())

from utils.constants import EventLabel, PostRetoldLabel, PreRetoldLabel  # noqa

with open(DATA_DIR / "coreference_extend.json") as f:
    corefs = json.load(f)
with open("splits.json") as f:
    splits = json.load(f)
labeled_pair_ids = splits["train_pair_ids"] + splits["val_pair_ids"] + splits["test_pair_ids"]
with open(DATA_DIR / "graphs.json") as f:
    graphs = json.load(f)

event_df = pd.read_csv(DATA_DIR / "event_entity.csv", index_col="index")
for c in node_types:
    event_df[f"{c}_token_ids"] = event_df[f"{c}_token_ids"].apply(json.loads)
    event_df[f"{c}_entity_ids"] = event_df[f"{c}_entity_ids"].apply(json.loads)
event_df["supports"] = event_df["supports"].apply(json.loads)
events = event_df.to_dict("index")

# Tokenize
tokenizer: LongformerTokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 1536
coref_token2longformer = {
    pair_id: {
        story_type: tokenizations.get_alignments(
            corefs[pair_id][story_type]["tokens"],
            [
                i[1:] if i.startswith("Ġ") else i
                for i in tokenizer.tokenize(corefs[pair_id][story_type]["story"])
            ],
        )
        for story_type in ["pre-retold", "post-retold"]
    }
    for pair_id in labeled_pair_ids
}
max_map_len = max(
    [
        len(
            list(
                chain.from_iterable(
                    [
                        coref_token2longformer[pair_id][story_type][0][i]
                        for i in entity_info["entity_token_ids"]
                    ]
                )
            )
        )
        for pair_id in labeled_pair_ids
        for story_type in ["pre-retold", "post-retold"]
        for _, entity_info in corefs[pair_id][story_type]["corefs"].items()
    ]
)

# Tokenize story
story_inputs = []
for pair_id, v in tqdm(graphs.items(), desc="Processing story"):
    for story_type, graph in v.items():
        model_input = tokenizer(corefs[pair_id][story_type]["story"], padding="max_length")
        assert (
            sum(model_input["attention_mask"])
            == len(coref_token2longformer[pair_id][story_type][1]) + 2
        )

        token_map = coref_token2longformer[pair_id][story_type][0]
        article_node2token = []
        for entity_id in range(len(graph)):
            entity_info = corefs[pair_id][story_type]["corefs"][str(entity_id)]
            # shift 1 for the CLS token
            lm_token_id = [
                i + 1
                for i in list(
                    chain.from_iterable([token_map[i] for i in entity_info["entity_token_ids"]])
                )
            ]
            assert lm_token_id
            raw = entity_info["entity"].lower().replace(" ", "")
            result = (
                tokenizer.decode(np.array(model_input["input_ids"])[lm_token_id])
                .lower()
                .replace(" ", "")
            )
            lm_token_id += [-1] * (max_map_len - len(lm_token_id))
            article_node2token.append(lm_token_id)

        story_inputs.append(
            {
                "pair_id": pair_id,
                "story_type": story_type,
                "graph": graph,
                "edge_index": from_networkx(nx.from_numpy_array(np.array(graph)))
                .cpu()
                .edge_index.numpy()
                .tolist(),
                "node2token": article_node2token,
                "input_ids": model_input["input_ids"],
                "mask": model_input["attention_mask"],
            }
        )
df = pd.DataFrame(story_inputs)
schema = pa.schema(
    [
        pa.field("pair_id", pa.string()),
        pa.field("story_type", pa.string()),
        pa.field("graph", pa.list_(pa.list_(pa.uint8()))),
        pa.field("edge_index", pa.list_(pa.list_(pa.uint8()), 2)),
        pa.field("input_ids", pa.list_(pa.uint16(), 1536)),
        pa.field("mask", pa.list_(pa.int8(), 1536)),
        pa.field("node2token", pa.list_(pa.list_(pa.int16(), max_map_len))),
    ]
)
table_write = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
pyarrow.parquet.write_table(table_write, DATA_DIR / "article.parquet")

# Tokenize event
max_nodes = max([len(vv) for v in graphs.values() for vv in v.values()])
max_nodes
inputs = []
for event_id, row in tqdm(event_df.iterrows(), total=len(event_df), desc="Processing event"):
    if row["story_type"] == "pre-retold":
        supports = row["supports"]
        NIR_label = PreRetoldLabel[row["event_type"]].value
    else:
        supports = row["supports"]
        NIR_label = PostRetoldLabel[row["event_type"]].value
    assert (
        row["event_type"] == EventLabel.additional.name
        or row["event_type"] == EventLabel.forgotten.name
        or supports
    )
    relate_node_idx = list(
        chain.from_iterable(
            [
                events[support][f"{c}_entity_ids"]
                for support in supports
                for c in node_types
                if events[support][f"{c}_entity_ids"]
            ]
        )
    )
    invert_story_type = "pre-retold" if row["story_type"] == "post-retold" else "post-retold"
    related_nodes = np.zeros(max_nodes, dtype=np.int8)
    related_nodes[len(graphs[row["pair_id"]][invert_story_type]) :] = -1
    if (
        row["event_type"] == EventLabel.consistent.name
        or row["event_type"] == EventLabel.inconsistent.name
    ):
        related_nodes[relate_node_idx] = 1

    model_input = tokenizer(row["event"])
    input_ids = model_input["input_ids"]
    mask = model_input["attention_mask"]
    node_ids = sorted(list(chain.from_iterable([row[f"{c}_entity_ids"] for c in node_types])))
    event_graph = np.array(graphs[row["pair_id"]][row["story_type"]])[node_ids][:, node_ids]
    triple = from_networkx(nx.from_numpy_array(np.array(event_graph))).cpu()
    triple_edge_index = triple.edge_index.tolist()

    inputs.append(
        {
            "pair_id": row["pair_id"],
            "story_type": row["story_type"],
            "event_id": event_id,
            "related_nodes": related_nodes,
            "input_ids": input_ids,
            "mask": mask,
            "node_ids": node_ids,
            "edge_index": triple_edge_index,
            "NIR_label": NIR_label,
        }
    )


input_df = pd.DataFrame(inputs)
schema = pa.schema(
    [
        pa.field("pair_id", pa.string()),
        pa.field("story_type", pa.string()),
        pa.field("event_id", pa.uint16()),
        pa.field("related_nodes", pa.list_(pa.int8(), max_nodes)),
        pa.field("input_ids", pa.list_(pa.uint16())),
        pa.field("mask", pa.list_(pa.uint8())),
        pa.field("node_ids", pa.list_(pa.uint8())),
        pa.field("edge_index", pa.list_(pa.list_(pa.uint8()), 2)),
        pa.field("NIR_label", pa.uint8()),
    ]
)
for split_type, pair_ids in splits.items():
    df = input_df[input_df["pair_id"].isin(pair_ids)]
    split_type = split_type.split("_")[0]
    table_write = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pyarrow.parquet.write_table(table_write, DATA_DIR / f"{split_type}.parquet")
    assert df[df["story_type"] == "pre-retold"]["NIR_label"].isin({0, 1}).all()
    assert df[df["story_type"] == "post-retold"]["NIR_label"].isin(PostRetoldLabel.labels()).all()
