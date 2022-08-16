import json
from pathlib import Path

DATA_DIR = Path(__file__).absolute().parent

nir = json.load(open(DATA_DIR / "NIR.json"))
tokens = json.load(open(DATA_DIR / "tokens.json"))


for event_id, value in nir.items():
    story_tokens = tokens[value["pair_id"]][value["story_type"]]
    story_tokens["-1"] = "{author}"
    for t in ["subject", "predicate", "object", "time"]:
        if t == "predicate" and value["explicitness"] == "implicit":
            continue
        if value[f"{t}_token_ids"]:
            value[t] = " ".join([story_tokens[str(i)] for i in value[f"{t}_token_ids"]])
        else:
            value[t] = None


json.dump(nir, open(DATA_DIR / "NIR_Hippocorpus.json", "w"))
