# Setup

```
conda create --name minDPR python=3.8
conda activate minDPR
pip install -r requirements.txt
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  # Need this if using A100s on CUDA 11
conda install -c pytorch faiss-gpu
conda deactivate
```

# Running a Pretrained DPR Retriever

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 encode_passages.py ../DPR/downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp 'data/psgs_w100_shard*.tsv' /data/local/minDPR_runs/DPR_pretrained/ --batch_size 2048 --num_workers 2 --gpus 0,1,2,3,4,5,6,7
```