import json
from pathlib import Path

import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
DATA_DIR = Path("../")


pair_ids = pd.read_json(DATA_DIR / "NIR.json").transpose()["pair_id"].unique()


def transform(df, type):
    result = df[(df["memType"] == type)].copy()

    return result.rename(
        {field: f"{type}_{field}" for field in ["mainEvent", "mostSurprising", "story", "summary"]},
        axis=1,
    )


df = pd.read_csv(DATA_DIR / "hippoCorpusV2.csv")
df = df.drop(index=4314)
df = df[
    ["mainEvent", "memType", "mostSurprising", "recAgnPairId", "recImgPairId", "story", "summary"]
]
imagined_df = transform(df, "imagined")
recalled_df = transform(df, "recalled")
retold_df = transform(df, "retold")
recalled_df = recalled_df[~pd.isnull(recalled_df["recAgnPairId"])]
retold_df = retold_df[~pd.isnull(retold_df["recAgnPairId"])]
df = recalled_df.merge(retold_df, on="recAgnPairId")
df = df[df["recAgnPairId"].isin(pair_ids)]
df = df.set_index("recAgnPairId")
df = df[
    [
        "recalled_summary",
        "recalled_story",
        "recalled_mainEvent",
        "recalled_mostSurprising",
        "retold_summary",
        "retold_story",
        "retold_mainEvent",
        "retold_mostSurprising",
    ]
]


error_df = pd.read_csv(DATA_DIR / "errors.csv")
for idx, row in error_df.iterrows():
    if row["pair_id"] not in pair_ids:
        print(row["pair_id"])
        continue

    df.loc[row["pair_id"], f'{row["story_type"]}_story'] = df.loc[
        row["pair_id"], f'{row["story_type"]}_story'
    ].replace(row["raw_piece"], row["corrected_piece"])


def parse_tokens(article):
    sents = sent_tokenize(article)
    tokens = {}
    count = 0
    for sent in sents:
        doc = nlp(sent)
        tokens.update({count + token.i: token.text for token in doc})
        count += len(list(doc))
    return tokens


tokens = {
    recAgnPairId: {
        "recalled": parse_tokens(row["recalled_story"]),
        "retold": parse_tokens(row["retold_story"]),
    }
    for recAgnPairId, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing")
}
articles = {
    recAgnPairId: {"recalled": row["recalled_story"], "retold": row["retold_story"]}
    for recAgnPairId, row in df.iterrows()
}
with open(DATA_DIR / "tokens.json", "w") as f:
    json.dump(tokens, f)

with open(DATA_DIR / "articles.json", "w") as f:
    json.dump(articles, f)
