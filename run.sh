#!/bin/bash
# chmod 777 run.sh
# ./run.sh 0,1,2,3,4,5,6,7
# ./run.sh 0,1,2,3,4,5,6,7 --seed 12345

OUTDIR="/data/local/minDPR_runs/nq"
model="${OUTDIR}/model"
pembs="${OUTDIR}/psgs_w100_shard*.pickle"
outfile="${OUTDIR}/out.json"

DATADIR="/common/home/jl2529/repositories/minDPR/data"
data_train="${DATADIR}/nq-train.json"
data_val="${DATADIR}/nq-dev.json"
data_test="${DATADIR}/nq-test.csv"
data_wiki_whole="${DATADIR}/psgs_w100.tsv"
data_wiki_shards="${DATADIR}/psgs_w100_shard*.tsv"

gpus=$1
commas="${gpus//[^,]}"
num_commas="${#commas}"
num_gpus="$((num_commas+1))"

optional1=${2:-}
optional2=${3:-}
optional3=${4:-}

train="torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus} train.py ${model} ${data_train} ${data_val} --num_warmup_steps 1237 --num_workers 2 --gpus ${gpus} ${optional1} ${optional2} ${optional3}"
encode="torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus} encode_passages.py ${model} '${data_wiki_shards}' ${OUTDIR} --batch_size 2048 --num_workers 2 --gpus ${gpus}"
search="python search.py ${model} ${data_test} '${pembs}' ${outfile} ${data_wiki_whole} --gpu 0"

echo $train
eval $train

echo $encode
eval $encode

echo $search
eval $search
