# Setup

```
conda create --name minDPR python=3.8
conda activate minDPR
pip install -r requirements.txt
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  # Need this if using A100s on CUDA 11
conda install -c pytorch faiss-gpu
conda deactivate
```

# DPR (Single NQ) Baseline

## Summary

|                                            | k=1             | k=5               | k=20            | k=100    |
| :---:                                      | :---:           | :---:             | :---:           | :---:    |
| Running DPR checkpoint using DPR code      | 46.4            | 68.6              | 80.1            | 86.1     |
| Running DPR checkpoint using this scode    | 46.5            | 68.6              | 80.1            | 86.1     |
| Evaluating released DPR result file        | 46.3            | 68.3              | 80.1            | 86.1     |
| DPR GitHub numbers: without hard negatives | 45.9            | 68.1              | 80.0            | 85.9     |
| DPR GitHub numbers: with hard negatives    | 52.5            | 72.2              | 81.3            | 87.3     |


## Commands

Running a DPR checkpoint
```
torchrun --standalone --nnodes=1 --nproc_per_node=8 encode_passages.py [DPR checkpoint] 'data/psgs_w100_shard*.tsv' [emb outdir] --batch_size 2048 --num_workers 2 --gpus 0,1,2,3,4,5,6,7  # ~30G, ~1h
python search.py [DPR checkpoint] data/nq-test.csv '[emb outdir]/psgs_w100_shard*_encoded.pickle' [outfile] data/psgs_w100.tsv --gpu 0  # RAM usage: ~90G after indexing
```

Evaluating a released result file
```
python evaluate.py [DPR test.json]
```
