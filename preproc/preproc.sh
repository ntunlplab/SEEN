gdown https://drive.google.com/uc?id=1tMokK7SkcMsCPUQy_WBsSizPSWQOQeZo -O splits.json
gdown https://drive.google.com/uc?id=11k2edFz9QrtMm-xg8DK_HirEDZZCwd8V -O not_split.tsv
python coreference.py
python preproc.py