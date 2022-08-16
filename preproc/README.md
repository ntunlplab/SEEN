# Steps
## 1. Download train/valid/test ids and the required file during preprocessing
```shell
gdown https://drive.google.com/uc?id=1tMokK7SkcMsCPUQy_WBsSizPSWQOQeZo -O splits.json
gdown https://drive.google.com/uc?id=11k2edFz9QrtMm-xg8DK_HirEDZZCwd8V -O not_split.tsv
```
## 2. Coreference Resolution
```shell
python coreference.py
```

## 3. Preprocessing the data
```shell
python preproc.py
```

