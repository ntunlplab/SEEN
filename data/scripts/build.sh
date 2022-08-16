gdown https://drive.google.com/uc?id=13F_9A8Z1jL9Eg4IwtRospfec7HQnubOC -O ../NIR.json
gdown https://drive.google.com/uc?id=1kaViqs9FDzArV_e8F7i7TZfeoEKkpRnc -O ../errors.csv
python -m spacy download en_core_web_sm
python tokenize_hippocorpus.py
python merge.py