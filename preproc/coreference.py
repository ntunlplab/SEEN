import json
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import tokenizations
from coreference_utils import get_clusters, load_model
from tqdm import tqdm

tqdm.pandas()
BASE_DIR = Path().absolute().parent
DATA_DIR = BASE_DIR / "data"


with open(DATA_DIR / "articles.json") as f:
    articles = json.load(f)


predictor = load_model()
nlp = spacy.load("en_core_web_trf")


with open("splits.json") as f:
    labeled_pair_ids = list(chain.from_iterable(json.load(f).values()))

# Coreference
results = {}
for pair_id in tqdm(labeled_pair_ids, desc="Coreference Resolution"):
    results[pair_id] = {}
    for story_type in ["pre-retold", "post-retold"]:
        story = articles[pair_id][story_type]
        doc = nlp(story)
        predict_result = predictor.predict(story)
        assert len(doc) == len(predict_result["document"])
        assert [i.text for i in doc] == predict_result["document"]
        clusters = get_clusters(doc, predict_result["clusters"])
        results[pair_id][story_type] = {
            "story": story,
            "tokens": [i.text for i in doc],
            "corefs": {
                entity_id: {
                    "entity": doc[entity[0].i : entity[-1].i + 1].text,
                    "entity_token_ids": [i.i for i in entity],
                    "mentions": {
                        mention_idx: {
                            "mention": doc[mention[0].i : mention[-1].i + 1].text,
                            "mention_token_ids": [i.i for i in mention],
                        }
                        for mention_idx, mention in enumerate(mentions)
                    },
                }
                for entity_id, (entity, mentions) in enumerate(clusters)
            },
        }


# find the nested entities, such as A entity = B entity + C entity
possessive_case = {"my", "your", "his", "her", "our", "their"}
for pair_id, v in results.items():
    for story_type, data in v.items():
        for entity_id, entity_info in data["corefs"].items():
            entity_info["subset"] = defaultdict(list)
            entity_info["subset_str"] = set()
            for mention_id, mentino_info in entity_info["mentions"].items():
                # compare to clusters
                for _entity_id, _entity_info in data["corefs"].items():
                    if _entity_id == entity_id:
                        continue
                    for _mention_id, _mentino_info in _entity_info["mentions"].items():
                        if _mentino_info["mention"].lower() in possessive_case:
                            continue
                        if set(_mentino_info["mention_token_ids"]).issubset(
                            (set(mentino_info["mention_token_ids"]))
                        ):
                            entity_info["subset"][_entity_id].append(_mention_id)
                            entity_info["subset_str"].add(_entity_info["entity"])

# split the nested entities
df = pd.DataFrame(
    [
        {"pair_id": pair_id, "story_type": story_type, "entity_id": entity_id, **entity_info}
        for pair_id, v in results.items()
        for story_type, data in v.items()
        for entity_id, entity_info in data["corefs"].items()
    ]
)
df["subset"] = df["subset"].apply(dict)
conditions = [
    df["subset"].apply(len) > 1,
    df["entity"].str.contains("and", case=False),
    df.apply(
        lambda row: len(
            {i.lower() for i in row["entity"].split()}
            - ({j.lower() for i in row["subset_str"] for j in i.split()} | {"and", "all", "both"})
        )
        > 1,
        axis=1,
    ),
]
need_split_df = df.copy(deep=True)[conditions[0] & conditions[1]]
need_split_df = need_split_df.set_index(["pair_id", "story_type", "entity_id"])
not_split_df = pd.read_csv("not_split.tsv", sep="\t", dtype=str)
not_split_df["used"] = False
not_split_df = not_split_df.set_index(["pair_id", "story_type", "entity_id"])
for pair_id, v in results.items():
    for story_type, data in v.items():
        for entity_id, entity_info in data["corefs"].items():
            index = (pair_id, story_type, entity_id)
            if index not in need_split_df.index or index in not_split_df.index:
                entity_info["subset"] = []
                entity_info["subset_str"] = []
                if index in not_split_df.index:
                    not_split_df.at[index, "used"] = True
            else:
                entity_info["subset"] = list(entity_info["subset"].keys())
                entity_info["subset_str"] = list(entity_info["subset_str"])
while True:
    nest_count = 0
    for pair_id, v in results.items():
        for story_type, data in v.items():
            for entity_id, entity_info in data["corefs"].items():
                if entity_info["subset"]:
                    nest_count += any(data["corefs"][i]["subset"] for i in entity_info["subset"])
    if nest_count == 0:
        break

    recompute = False
    for pair_id, v in results.items():
        for story_type, data in v.items():
            for entity_id, entity_info in data["corefs"].items():
                if not entity_info["subset"]:
                    continue

                removed_idx = subset = None
                for idx, i in enumerate(entity_info["subset"]):
                    if data["corefs"][i]["subset"]:
                        removed_idx = idx
                        subset = data["corefs"][i]["subset"]
                        break
                if removed_idx is not None:
                    entity_info["subset"] = (
                        entity_info["subset"][:removed_idx]
                        + entity_info["subset"][removed_idx + 1 :]
                    )
                    entity_info["subset"] = list(set(entity_info["subset"] + subset))
                    recompute = True
                    break
            if recompute:
                break
        if recompute:
            break
# transform key to string type
with open(DATA_DIR / "coreference.json", "w") as f:
    json.dump(results, f)
with open(DATA_DIR / "coreference.json") as f:
    results = json.load(f)

# merge coreference resolution results and events
node_types = ["subject", "predicate", "object"]
corefs = results
event_df = pd.read_json(DATA_DIR / "NIR_Hippocorpus.json").transpose()
with open(DATA_DIR / "tokens.json") as f:
    tokens = json.load(f)
    tokens = {
        pair_id: {story_type: [ts[str(i)] for i in range(len(ts))] for story_type, ts in v.items()}
        for pair_id, v in tokens.items()
    }


def get_text(entity):
    if pd.isnull(entity):
        return None
    elif entity == "{author}":
        return "I"
    else:
        return entity


event_df["event"] = event_df.apply(
    lambda row: " ".join([get_text(row[c]) for c in node_types if get_text(row[c])])
    .strip()
    .replace("  ", " "),
    axis=1,
)


# implicit predicate
implicit_predicate_count = defaultdict(lambda: defaultdict(lambda: 1))
for event_id, row in event_df.iterrows():
    if pd.isnull(row["predicate"]) or row["predicate_token_ids"]:
        continue

    pair_id = row["pair_id"]
    story_type = row["story_type"]
    idx = implicit_predicate_count[pair_id][story_type]
    event = f"Event{idx}: {row['event']}."
    event_tokens = [token.text for token in nlp(event)]

    predicate_token_idices = tokenizations.get_alignments(
        event_tokens,
        [
            i
            for i in (
                f"Event{idx}:",
                row["subject"] if not pd.isnull(row["subject"]) else "",
                row["predicate"] if not pd.isnull(row["predicate"]) else "",
                row["object"] if not pd.isnull(row["object"]) else "",
                ".",
            )
        ],
    )[1][2]

    corefs[pair_id][story_type]["story"] += f" {event}"
    corefs[pair_id][story_type]["tokens"] += event_tokens
    for event_token_idx, token in enumerate(event_tokens):
        if event_token_idx in predicate_token_idices:
            event_df.at[event_id, "predicate_token_ids"].append(
                len(tokens[pair_id][story_type]) + event_token_idx
            )
    tokens[pair_id][story_type] += event_tokens

# align spacy tokenization and coreference resolution tokens
token_alignment = {
    pair_id: {
        story_type: tokenizations.get_alignments(
            spacy_tokens, corefs[pair_id][story_type]["tokens"]
        )[0]
        for story_type, spacy_tokens in v.items()
    }
    for pair_id, v in tokens.items()
    if pair_id in labeled_pair_ids
}
possessive_case = ["my", "your", "his", "her", "our", "their", "its", "whose", "whosever"]
self_entity_special_case = {
    "3RKNTXVS3NKSI9BOEGJK201GEQRA4N": {"post-retold": "We"},
    "3B3WTRP3DCO4ACMWIWMWDQTYHR692R": {"pre-retold": "We"},
    "39RP059MEIFD595MQYJCJ52M8O2MBY": {"post-retold": "We"},
    "39GAF6DQWSMVIYH32TRE0P8QMTWV1Z": {"post-retold": "us as a family"},
    "30JNVC0ORA6EH160IJ0PMPPPOFBQHD": {"post-retold": "we"},
    "3907X2AHF1RP2Z23ZLITZGTDSEOP2A": {"pre-retold": "CS and I"},
    "3QBD8R3Z225HKD5POZO23VLL37TO4Y": {"post-retold": "We"},
    "33M4IA01QHNBFSLF027BU1NOZS9RXE": {"pre-retold": "We"},
    "3CN4LGXD5YATER9RUMX05MJNF0NY4E": {"post-retold": "we"},
    "3SB4CE2TJWHJGBZYYRLPZBYE7LUAX0": {"pre-retold": "Grandma"},
    "3QL2OFSM9742XWISGZU774X44NQCNU": {"pre-retold": "Mom", "post-retold": "Mom"},
    "3OB0CAO74IBNQ3XM9THJZBSRMNYYHR": {"pre-retold": "we"},
    "39OWYR0EPLD3C76GE3TJWQGE79LYFW": {"post-retold": "We"},
    "3WQ3B2KGE92G9KJXXC0EZDEJI7WB10": {
        "pre-retold": "The group of us guys that have hung out together over the last 10 years"
    },
    "3O7L7BFSHFBPGTRFFANASK1QUPQEIV": {"post-retold": "we"},
    "3KAKFY4PGVOM6VBIQQ6E9TXTDYY3IX": {"pre-retold": "We"},
    "31QTRG6Q2UZF3KVAS6PO8KUI8DLPYY": {"post-retold": "we"},
    "3IXEICO7935BCSEB343GU3BMY6F6TM": {"pre-retold": "son"},
    "3Y4W8Q93L06PATXFIMB91R9ZDJ0VDE": {"post-retold": "We"},
    "3HWRJOOET6OEA7UBKMHOWM7DEN0ES6": {"pre-retold": "I"},
    "3LO69W1SU4ZPQ8VC68ABK3EUZ97GLB": {"pre-retold": "Dad"},
}
for c in node_types:
    event_df[f"{c}_entity_id"] = None

# replace {author} entity with the entity which is author of the story
invalid = defaultdict(lambda: defaultdict(dict))
for event_id, row in event_df.iterrows():
    pair_id = row["pair_id"]
    story_type = row["story_type"]
    for c in node_types:
        if pd.isnull(row[c]):
            continue

        if row[f"{c}_token_ids"] == [-1]:
            if (
                len(
                    self_entity := [
                        entity_id
                        for entity_id, data in corefs[pair_id][story_type]["corefs"].items()
                        if (data["entity"].lower() in {"i", "my", "mine", "me"})
                    ]
                )
                == 1
            ):
                event_df.loc[event_id, f"{c}_entity_id"] = int(self_entity[0])
            elif (
                len(
                    self_entity := [
                        entity_id
                        for entity_id, data in corefs[pair_id][story_type]["corefs"].items()
                        if {
                            mention_id
                            for mention_id, data in data["mentions"].items()
                            if data["mention"].lower() in {"i"}
                        }
                    ]
                )
                == 1
            ):
                event_df.loc[event_id, f"{c}_entity_id"] = int(self_entity[0])
            elif (
                story_type in self_entity_special_case.get(pair_id, {})
                and len(
                    self_entity := [
                        entity_id
                        for entity_id, data in corefs[pair_id][story_type]["corefs"].items()
                        if data["entity"] == self_entity_special_case[pair_id][story_type]
                    ]
                )
                == 1
            ):
                event_df.loc[event_id, f"{c}_entity_id"] = int(self_entity[0])
            elif (
                pair_id == "3EFVCAY5L4V4231UKPMACBK471PJ8W"
                and story_type == "post-retold"
                and len(
                    self_entity := [
                        entity_id
                        for entity_id, data in corefs[pair_id][story_type]["corefs"].items()
                        if data["entity"].lower() == "i"
                        and any(
                            mention_info["mention"].lower() == "me"
                            for mention_info in data["mentions"].values()
                        )
                    ]
                )
                == 1
            ):
                event_df.loc[event_id, f"{c}_entity_id"] = int(self_entity[0])
            else:
                invalid[pair_id][story_type]["entities"] = [
                    data["entity"] for _, data in corefs[pair_id][story_type]["corefs"].items()
                ]
            continue

        converted_token_ids = set(
            chain.from_iterable(
                [token_alignment[pair_id][story_type][i] for i in row[f"{c}_token_ids"]]
            )
        )
        for entity_id, entity_info in corefs[pair_id][story_type]["corefs"].items():
            for mention_idx, mention_info in entity_info["mentions"].items():
                mention_token_ids = set(mention_info["mention_token_ids"])
                if converted_token_ids == mention_token_ids:
                    assert event_df.loc[event_id, f"{c}_entity_id"] is None
                    event_df.at[event_id, f"{c}_entity_id"] = int(entity_id)
assert not invalid


# add new entities for the nodes cannot matched the original coreference resolution results
for pair_id in tqdm(labeled_pair_ids, desc="Adding new entities"):
    for story_type in ["pre-retold", "post-retold"]:
        sub_df = event_df[(event_df["pair_id"] == pair_id) & (event_df["story_type"] == story_type)]
        assert set(int(i) for i in corefs[pair_id][story_type]["corefs"].keys()) == set(
            range(len(corefs[pair_id][story_type]["corefs"]))
        )

        additional_entity = {}
        max_id = max([int(i) for i in corefs[pair_id][story_type]["corefs"]])
        counter = 1
        for idx, row in sub_df.iterrows():
            for c in node_types:
                if row[f"{c}_entity_id"] is not None or pd.isnull(row[c]):
                    continue

                for entity_id, token_ids in additional_entity.items():
                    if token_ids == row[f"{c}_token_ids"]:
                        event_df.at[idx, f"{c}_entity_id"] = int(entity_id)

                if not pd.isnull(event_df.at[idx, f"{c}_entity_id"]):
                    continue

                new_id = max_id + counter
                event_df.at[idx, f"{c}_entity_id"] = new_id
                assert str(new_id) not in corefs[pair_id][story_type]["corefs"]
                additional_entity[str(new_id)] = row[f"{c}_token_ids"]
                converted_token_ids = set(
                    chain.from_iterable(
                        [token_alignment[pair_id][story_type][i] for i in row[f"{c}_token_ids"]]
                    )
                )
                assert converted_token_ids
                corefs[pair_id][story_type]["corefs"][str(new_id)] = {
                    "entity": row[c],
                    "entity_token_ids": sorted(list(converted_token_ids)),
                    "mentions": {
                        "0": {
                            "mention": row[c],
                            "mention_token_ids": sorted(list(converted_token_ids)),
                        }
                    },
                    "subset": [],
                }
                counter += 1
        assert set(int(i) for i in corefs[pair_id][story_type]["corefs"].keys()) == set(
            range(len(corefs[pair_id][story_type]["corefs"]))
        )

# remove useless entities
new_corefs = deepcopy(corefs)
for pair_id, v in tqdm(new_corefs.items(), desc="Removing useless entities"):
    for story_type, data in v.items():
        data["corefs"] = {}

        sub_df = event_df[(event_df["pair_id"] == pair_id) & (event_df["story_type"] == story_type)]
        used_entity_ids = set(
            int(i)
            for node_type in node_types
            for i in sub_df[f"{node_type}_entity_id"]
            if not pd.isnull(i)
        ) | set(
            int(j)
            for node_type in node_types
            for i in sub_df[f"{node_type}_entity_id"]
            for j in (
                [] if pd.isnull(i) else corefs[pair_id][story_type]["corefs"][str(i)]["subset"]
            )
        )
        sort_df = pd.DataFrame(
            [
                {
                    "entity_id": int(entity_id),
                    "entity": entity_info["entity"],
                    "not_used": int(entity_id) not in used_entity_ids,
                    "subset_len": len(entity_info["subset"]),
                }
                for entity_id, entity_info in corefs[pair_id][story_type]["corefs"].items()
            ]
        )

        new2old = (
            sort_df.sort_values(["not_used", "subset_len"]).reset_index()["entity_id"].to_dict()
        )
        old2new = {v: k for k, v in new2old.items()}
        ori_data = deepcopy(corefs[pair_id][story_type])
        for new_id, old_id in new2old.items():
            data["corefs"][str(new_id)] = deepcopy(ori_data["corefs"][str(old_id)])
            data["corefs"][str(new_id)]["subset"] = [
                str(old2new[int(i)]) for i in ori_data["corefs"][str(old_id)]["subset"]
            ]
        for idx, row in sub_df.iterrows():
            for c in node_types:
                if pd.isnull(event_df.at[idx, f"{c}_entity_id"]):
                    continue
                event_df.at[idx, f"{c}_entity_id"] = int(
                    old2new[event_df.at[idx, f"{c}_entity_id"]]
                )
with open(DATA_DIR / "coreference_extend.json", "w") as f:
    json.dump(new_corefs, f)
corefs = new_corefs

# save entity ids of each node
for c in node_types:
    event_df[f"{c}_entity_ids"] = event_df.apply(lambda x: list(), axis=1)
for pair_id in tqdm(labeled_pair_ids, desc="Saving"):
    for story_type in ["pre-retold", "post-retold"]:
        sub_corefs = corefs[pair_id][story_type]["corefs"]
        sub_df = event_df[(event_df["pair_id"] == pair_id) & (event_df["story_type"] == story_type)]
        for idx, row in sub_df.iterrows():
            for c in node_types:
                if pd.isnull(row[c]):
                    event_df.at[idx, f"{c}_entity_ids"] = []
                    continue

                event_df.at[idx, f"{c}_entity_ids"] = [
                    int(i) for i in sub_corefs[str(row[f"{c}_entity_id"])]["subset"]
                ] or [event_df.at[idx, f"{c}_entity_id"]]

        sub_df = event_df[(event_df["pair_id"] == pair_id) & (event_df["story_type"] == story_type)]
        used_entity_ids = set(
            int(j) for c in node_types for i in sub_df[f"{c}_entity_ids"].values for j in i
        )
        assert set(range(len(used_entity_ids))) == used_entity_ids
        used_entity_ids = set(
            int(j) for c in node_types for i in sub_df[f"{c}_entity_ids"].values for j in i
        ) | set(
            int(i) for c in node_types for i in sub_df[f"{c}_entity_id"].values if not pd.isnull(i)
        )
        assert set(range(len(used_entity_ids))) == used_entity_ids
event_df.index.rename("index", inplace=True)
event_df.to_csv(DATA_DIR / "event_entity.csv")

# save the adjacency matrix of each node
graphs = {}
for pair_id in tqdm(labeled_pair_ids, desc="Saving adjacency matrix"):
    graphs[pair_id] = {}
    for story_type in ["pre-retold", "post-retold"]:
        sub_df = event_df[(event_df["pair_id"] == pair_id) & (event_df["story_type"] == story_type)]

        used_entity_ids = set(
            int(j) for c in node_types for i in sub_df[f"{c}_entity_ids"].values for j in i
        )
        matrix = np.zeros((len(used_entity_ids), len(used_entity_ids)), dtype=int)
        for _, row in sub_df.iterrows():
            for i in row["subject_entity_ids"]:
                for j in row["predicate_entity_ids"]:
                    matrix[i, j] = matrix[j, i] = 1
            if not pd.isnull(row["object"]):
                for i in row["predicate_entity_ids"]:
                    for j in row["object_entity_ids"]:
                        matrix[i, j] = matrix[j, i] = 1
        assert matrix.sum(axis=0).all() and matrix.sum(axis=1).all()
        graphs[pair_id][story_type] = matrix.tolist()
with open(DATA_DIR / "graphs.json", "w") as f:
    json.dump(graphs, f)
