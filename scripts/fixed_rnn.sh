#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/trainer.py \
    --output_dir="output" \
    --data_path="data/ptb/ptb.pkl" \
    --network_type="rnn" \
    --fixed_arc='0 2 1 0 2 1 2 2 4 0 5 0 3 2 6 2' \

