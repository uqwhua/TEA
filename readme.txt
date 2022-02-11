##Dependencies

Python 3
PyTorch >= 1.0
Scikit Learn
huggingface/transformers == 1.1.0


##Code

Run the following script to train all the subgraphs.
# Example
python -u dev_train_subgraph.py --gpu_id=-1 --dataset='yg_dbp' --datadir='DWY100k' --hard=0 --channel all

Run the following script for ensemble.
# Example
python -u dev_ensemble_subgraphs.py --gpu_id=-1 --dataset='yg_dbp' --datadir='DWY100k'

channel: {Digital, Literal, Structure, Name, Time, All}
dataset: {zh_en, fr_en, ja_en, wd_dbp, yg_dbp}
datadir: {DBP15k, DWY100k}

Download the Word2Vec file for initializing the corresponding name/relation/attributes embeddings.


##Datasets

Download the original datasets from https://1drv.ms/u/s!AuQRz5abAH5T2jDOmiMlkqFP8s0Z?e=V6wNWS

Download the hard setting datasets from https://1drv.ms/u/s!AuQRz5abAH5T3EWhCpZrw24jTOrm?e=ufjzfW

