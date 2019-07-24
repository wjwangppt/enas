#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/trainer.py \
    --output_dir="output" \
    --data_path="data/ptb/ptb.pkl" \
    --network_type="rnn" \
    --rnn_num_layers=8 \
