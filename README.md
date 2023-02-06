# TEA: Time-aware Entity Alignment in Knowledge Graphs

The code of our WWW2023 paper TEA: Time-aware Entity Alignment in Knowledge Graphs

## Environment and Dependencies
* Python 3
* [PyTorch >= 1.0](https://pytorch.org/get-started/locally/)
* [Scikit Learn](https://scikit-learn.org/stable/)
* [huggingface/transformers == 1.1.0](https://github.com/huggingface/transformers)
* torch-geometric

create a new environment
```bash
conda create --name exp python=3.9

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# install BERT
conda install -c huggingface transformers

# other tools
pip install -U scikit-learn
pip install torch-geometric
conda install pytorch-sparse -c pyg
```

## Results

1. Run the following scripts to train the alignment model on all kinds of information.
```bash
# train
python train.py --gpu_id 0 --channel all --dataset DBP15k/ja_en --load_hard_split 

# evaluate
python evaluate.py --gpu_id 0 --dataset DBP15k/ja_en --load_hard_split 
```
\
Channels: {Digital, Literal, Structure, Name, Time}

Datasets: {DBP15k/ja_en, DBP15k/fr_en, DBP15k/zh_en, DWY100k/wd_dbp, DWY100k/yg_dbp}

## Datasets
Download all the datasets from [OneDrive](https://1drv.ms/u/s!AnRJvk5zSd0HgXKsFQ1vEaYxO7FX?e=2MnJyP) and unzip it under the current folder.



