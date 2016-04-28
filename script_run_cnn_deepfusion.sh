#!/usr/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_h5> <input_json> [gpuid=0]"
    exit
fi

h5_fn=$1
json_fn=$2

gpuid=0

if [ $# -ge 3 ]; then
    gpuid=$3
fi

CUDA_VISIBLE_DEVICES=$gpuid th train_cnn_deepfusion.lua -input_h5 $h5_fn -input_json $json_fn -gpuid 0  -losses_log_every 100 -rnn_size 512
