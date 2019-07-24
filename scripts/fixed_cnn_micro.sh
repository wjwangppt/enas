#!/bin/bash

export PYTHONPATH="$(pwd)"

fixed_arc="0 1 1 0 0 2 0 0 0 0 3 0 0 1 0 4 0 1 1 4"
fixed_arc="$fixed_arc 1 3 1 0 1 4 1 3 1 0 1 0 1 1 1 2 1 0 1 4"

python src/trainer.py \
    --output_dir="output" \
    --data_path="data/cifar10" \
    --network_type="cnn" \
    --search_for="micro" \
    --controller_training=FALSE \
    --cnn_micro_num_layers=15 \
    --cnn_micro_num_cells=5 \
    --batch_size=128 \
    --num_epochs=630 \
    --child__fixed_arc="${fixed_arc}" \
