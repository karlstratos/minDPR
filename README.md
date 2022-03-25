This is minimal PyTorch code to replicate single NQ retrieval results.

# Setup

```
conda create --name minDPR python=3.8
conda activate minDPR
pip install -r requirements.txt
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  # Need this if using A100s on CUDA 11
conda install -c pytorch faiss-gpu
conda deactivate
```
For simplicity, the code assumes that all input files are in `data/`, specifically `nq-train.json`, `nq-dev.json`, `nq-test.csv`, and `psgs_w100.tsv`.
Shard the passage file by calling
```
python shard_wiki.py data/psgs_w100.tsv --num_shards 10
```
Modify the data directory and output directory paths in the `run.sh` script.


# Training

## Results

|                                                     | k=1             | k=5               | k=20            | k=100    |
| :---:                                               | :---:           | :---:             | :---:           | :---:    |
| minDPR (seed 42)                                    | 46.3            | 68.4              | 79.5            | 86.2     |
| minDPR (seed 12345)                                 | 45.5            | 68.9              | 79.7            | 86.0     |
| minDPR + DPR dataloader + pad_to_max (seed 42)      | 46.6            | 68.1              | 79.8            | 86.2     |
| minDPR + DPR dataloader + pad_to_max (seed 12345)   | 46.0            | 68.2              | 79.9            | 86.4     |
| DPR GitHub code                                     | 46.0            | 68.2              | 79.1            | 86.3     |

Observations
 - Training using the official DPR GitHub code (i.e., last row) is the most fair baseline. minDPR matches it (e.g., first row).
 - Some variance using different random seeds.


## Commands

```
./mindpr/run.sh 0,1,2,3,4,5,6,7
```
or explicitly
```
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py /data/local/minDPR_runs/nq/model data/nq-train.json data/nq-dev.json --num_warmup_steps 1237 --num_workers 2 --gpus 0,1,2,3,4,5,6,7  # 20-37G, 1h 54m, best epoch 37
torchrun --standalone --nnodes=1 --nproc_per_node=8 encode_passages.py /data/local/minDPR_runs/nq/model 'data/psgs_w100_shard*.tsv' /data/local/minDPR_runs/nq --batch_size 2048 --num_workers 2 --gpus 0,1,2,3,4,5,6,7  # 29-39G, 1h 4m
python search.py /data/local/minDPR_runs/nq/model data/nq-test.csv '/data/local/minDPR_runs/nq/psgs_w100_shard*.pickle' /data/local/minDPR_runs/nq/out.json data/psgs_w100.tsv --gpu 0  # 11m
```

# Pretrained/Reported DPR Results

## Results

|                                            | k=1             | k=5               | k=20            | k=100    |
| :---:                                      | :---:           | :---:             | :---:           | :---:    |
| Running DPR checkpoint using DPR code      | 46.4            | 68.6              | 80.1            | 86.1     |
| Running DPR checkpoint using this scode    | 46.5            | 68.6              | 80.1            | 86.1     |
| Evaluating released DPR result file        | 46.3            | 68.3              | 80.1            | 86.1     |
| DPR GitHub numbers: without hard negatives | 45.9            | 68.1              | 80.0            | 85.9     |
| DPR GitHub numbers: with hard negatives    | 52.5            | 72.2              | 81.3            | 87.3     |
| DPR paper (2020)                           | ---             | ---               | 78.4            | 85.4     |

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
