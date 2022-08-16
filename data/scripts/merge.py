import json
from pathlib import Path

DATA_DIR = Path("../")

nir = json.load(open(DATA_DIR / "NIR.json"))
tokens = json.load(open(DATA_DIR / "tokens.json"))


story_map = {"pre-retold": "recalled", "post-retold": "retold"}
for event_id, value in nir.items():
    story_tokens = tokens[value["pair_id"]][story_map[value["story_type"]]]
    story_tokens["-1"] = "{author}"
    for t in ["subject", "predicate", "object", "time"]:
        if t == "predicate" and value["explicitness"] == "implicit":
            continue
        value[t] = " ".join([story_tokens[str(i)] for i in value[f"{t}_token_ids"]])


json.dump(nir, open(DATA_DIR / "NIR_Hippocorpus.json", "w"))
