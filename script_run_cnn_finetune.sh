#!/usr/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_fn> <input_h5> <input_json> [gpuid=1]"
    exit
fi

model_fn=$1
h5_fn=$2
json_fn=$3

gpuid=1

if [ $# -ge 4 ]; then
    gpuid=$4
fi

CUDA_VISIBLE_DEVICES=$gpuid th train_cnn.lua -start_from $model_fn -input_h5 $h5_fn -input_json $json_fn -gpuid 0  -losses_log_every 100 -rnn_size 512 -finetune_cnn_after 0 -batch_size 10
