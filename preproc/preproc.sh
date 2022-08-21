gdown https://drive.google.com/uc?id=1tMokK7SkcMsCPUQy_WBsSizPSWQOQeZo -O splits.json
gdown https://drive.google.com/uc?id=11k2edFz9QrtMm-xg8DK_HirEDZZCwd8V -O not_split.tsv
python -m spacy download en_core_web_trf
python coreference.py
python preproc.py