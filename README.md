# TEA: Time-aware Entity Alignment in Knowledge Graphs
Source code of paper "TEA: Time-aware Entity Alignment in Knowledge Graphs", which has been accepted by TheWebConf'2023.

## Environment and Dependencies
* Python 3
* [PyTorch >= 1.0](https://pytorch.org/get-started/locally/)
* [Scikit Learn](https://scikit-learn.org/stable/)
* [huggingface/transformers == 1.1.0](https://github.com/huggingface/transformers)
* [torch-geometric](https://github.com/pyg-team/pytorch_geometric)

### Create a new environment
```bash
>> conda create --name exp python=3.9
# Install pytorch
>> conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# Install BERT
>> conda install -c huggingface transformers
# Other tools
>> pip install -U scikit-learn
>> pip install torch-geometric
>> conda install pytorch-sparse -c pyg
```

## Datasets
Download the original DBP15K and DWY100K datasets from [here](https://1drv.ms/u/s!AuQRz5abAH5T2jDOmiMlkqFP8s0Z?e=V6wNWS).

Download the [hard setting](https://aclanthology.org/2020.emnlp-main.515/) DBP15K datasets from [here](https://1drv.ms/u/s!AuQRz5abAH5T3EWhCpZrw24jTOrm?e=ufjzfW).

## Run the TEA model
Run the following scripts to train and evaluate the TEA model on all kinds of information.
```bash
# Train
>> python train.py --gpu_id 0 --channel all --dataset DBP15k/ja_en --load_hard_split 
# Evaluate
>> python evaluate.py --gpu_id 0 --dataset DBP15k/ja_en --load_hard_split 
```
Channels: {Digital, Literal, Structure, Name, Time, All}

Datasets: {DBP15k/ja_en, DBP15k/fr_en, DBP15k/zh_en, DWY100k/wd_dbp, DWY100k/yg_dbp}

Download the [Word2Vec file](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) for initializing the corresponding name/relation/attribute embeddings.

## Citation
If you find our TEA model and the experimental results useful, please kindly cite the following paper:
```
@inproceedings{liu2023tea,
  author = {Liu, Yu and Hua, Wen and Xin, Kexuan and Hosseini, Saeid and Zhou, Xiaofang},
  title = {TEA: Time-aware Entity Alignment in Knowledge Graphs},
  booktitle = {Proceedings of the ACM Web Conference 2023},
  series = {WWW'23},
  pages = {},
  location = {Austin, Texas, USA},
  year = {2023},
  doi = {10.1145/3543507.3583317}
}
```

## Acknowledgement
We used the code of these models: [RDGCN](https://github.com/StephanieWyt/RDGCN), [DGMC](https://github.com/rusty1s/deep-graph-matching-consensus), [KEGCN](https://github.com/PlusRoss/KE-GCN), [RREA](https://github.com/MaoXinn/RREA), [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [PSR](https://github.com/MaoXinn/PSR), [AttrE](https://bitbucket.org/bayudt/kba/src/master/), [MultiKE](https://github.com/nju-websoft/MultiKE), [AttrGNN](https://github.com/thunlp/explore-and-evaluate), [TEA-GNN](https://github.com/soledad921/TEA-GNN). Thanks for the authors' great contributions!
