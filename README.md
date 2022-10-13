# TEA

Source code of paper "TEA: Time-aware Entity Alignment in Knowledge Graphs", which has been submitted to WebConf'2023.

## Dependencies

* Python 3
* [PyTorch >= 1.0](https://pytorch.org/get-started/locally/)
* [Scikit Learn](https://scikit-learn.org/stable/)
* [huggingface/transformers == 1.1.0](https://github.com/huggingface/transformers)

## Code

Step 1. Run the following script to train all the subgraphs.
```bash
# Example
>> python -u dev_train_subgraph.py --gpu_id=-1 --dataset='yg_dbp' --datadir='DWY100k' --hard=0 --channel all
```

Step 2. Run the following script for ensemble.
```bash
# Example
>> python -u dev_ensemble_subgraphs.py --gpu_id=-1 --dataset='yg_dbp' --datadir='DWY100k'
```

Channel: {Digital, Literal, Structure, Name, Time, All}

Dataset: {zh_en, fr_en, ja_en, wd_dbp, yg_dbp}

Datadir: {DBP15k, DWY100k}

Download the [Word2Vec file](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) for initializing the corresponding name/relation/attributes embeddings.


## Datasets

Download the original DBP15K and DWY100K datasets from [here](https://1drv.ms/u/s!AuQRz5abAH5T2jDOmiMlkqFP8s0Z?e=V6wNWS).

Download the [hard setting](https://aclanthology.org/2020.emnlp-main.515/) DBP15K datasets from [here](https://1drv.ms/u/s!AuQRz5abAH5T3EWhCpZrw24jTOrm?e=ufjzfW).
