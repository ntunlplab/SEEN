NIR
===

# Introduction

Hippocorpus is constructed for investigating the difference in the narrative flow between relating life experiences and telling imaginative stories.
We construct **NIR** by pruning the imaginative stories in Hippocorpus and retaining those stories about real-life events written by crowdworkers at two different times as pre-retold stories and post-retold stories.
We summarize the following five event types from the story pairs in the dataset: **Consistent**, **Inconsistent**, **Additional**, **Forgotten**, and **Unforgotten**.

# Format

Each object of the JSON files is consisted of event_id(i.e., object key), pair_id, story_type, subject, predicate, object, time, event_type, and the support evidences of the event.

# Example

```json
{
  "59": {
    "pair_id": "3P4RDNWND6SXR9D7TBY1P0EI0KHJIR",
    "story_type": "post-retold",
    "explicitness": "explicit",
    "subject_token_ids": [
      24,
      25,
      26,
      27,
      28,
      29,
      30
    ],
    "predicate": null,
    "predicate_token_ids": [
      31,
      32,
      33
    ],
    "object_token_ids": [
      35
    ],
    "time_token_ids": [],
    "event_type": "additional",
    "supports": []
  },
  ...
}
```

# Steps

## 1. Download the corpus--Hippocorpus
Since we construct our dataset--NIR by exteding the Hippocorpus, we need to download the hippocorpus first. 
1. Go to [http://aka.ms/hippocorpus](http://aka.ms/hippocorpus)
2. Login your Microsoft accouot.
3. Download `hippoCorpusV2.csv` and save it to the parent directory(i.e. `data/`).

## 2. Download NIR dataset & Hippocorpus correction file
```shell
gdown https://drive.google.com/uc?id=13F_9A8Z1jL9Eg4IwtRospfec7HQnubOC -O ../NIR.json
gdown https://drive.google.com/uc?id=1kaViqs9FDzArV_e8F7i7TZfeoEKkpRnc -O ../errors.csv
```

## 3. Download spacy model
We use spacy tok tokenize the stories. Thus, we need to download the spacy model.
```sh
python -m spacy download en_core_web_sm
```

## 4. Tokenize the Stories in Hippocorpus
Since we only release the annotation that uses the tokenized result of the Hippocorpus, we provide the script for tokenization and preprocessing to ensure the result is the same as ours.
```sh
python tokenize_hippocorpus.py
```

## 5. Merge NIR and Hippocorpus
After parsing the Hipporcorpus, we also provide the script to merge the hippocorpus and the NIR for convenience.
```sh
python merge.py
```