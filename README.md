SEEN: Structured Event Enhancement Network for Explainable Need Detection of Information Recall Assistance
===

This repo provides the source code & data of our paper SEEN: Structured Event Enhancement Network for Explainable Need Detection of Information Recall Assistance. If you use any of our code, processed data or pretrained models, please cite:
```bib
```

# Dependency
- OS: Linux/Debian
- Python: 3.9.10
- Pytorch: 1.11.0
- CUDA: 11.5

Command of package installation
```shell
pip install torch==1.11.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu115.html
pip install -r requirements.txt
# cached-path > 1.1.2 has below bug 
# AttributeError: module 'cached_path' has no attribute 'file_friendly_logging'
pip install cached-path==1.1.2
```

# Data Processing
#### Please refer to the `data/script` folder for data constrcion scripts.
#### Please refer to the `preproc` folder for data preprocessing scripts.

# SEEN Usage
We pre-define four experiments setting for coonvenience.
- SEENLongformer
- SEENLongformerLarge
## Train
```shell
python main.py \
    --do_train \
    --exp_name=$exp \
    --batch_size=$BATCH_SIZE \
    --val_batch_size=$VAL_BATCH_SIZE \
    --gpus=$GPUS \
    --val_step=$VAL_STEP \
    --epochs=3 \
    --seed=$seed \
    --pretrained_path=$seed # pass this vvalue to utilize the pretrained model)
```

## Validate
```shell
python main.py \
    --do_val \
    --exp_name=$exp \
    --val_batch_size=$VAL_BATCH_SIZE
```

## Test
```shell
python main.py \
    --do_test \
    --exp_name=$exp \
    --val_batch_size=$VAL_BATCH_SIZE
    --test_model_path=$TEST_MODEL_PATH # pass this value to test specific checkpoint)
```

## Model Checkpoints
- [SEENLongformer](https://drive.google.com/file/d/1l3r0kR79PoNUgUAjLVcTya_-XSxJSRyf/view?usp=sharing)
- [SEENLongformerLarge](https://drive.google.com/file/d/1CEORS_HeZ5kdxVUktUS6iK8_928eDHGG/view?usp=sharing)