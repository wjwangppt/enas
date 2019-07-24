#!/bin/bash

export PYTHONPATH="$(pwd)"

fixed_arc="2"
fixed_arc="$fixed_arc 4 0"
fixed_arc="$fixed_arc 4 1 1"
fixed_arc="$fixed_arc 4 1 1 0"
fixed_arc="$fixed_arc 5 1 1 0 0"
fixed_arc="$fixed_arc 0 1 1 0 1 1"
fixed_arc="$fixed_arc 0 1 0 0 1 0 0"
fixed_arc="$fixed_arc 3 0 0 1 0 1 0 0"
fixed_arc="$fixed_arc 1 0 0 0 0 0 0 1 0"
fixed_arc="$fixed_arc 5 0 1 0 0 0 1 0 1 1"
fixed_arc="$fixed_arc 3 1 1 1 0 1 0 0 0 1 1"
fixed_arc="$fixed_arc 4 1 0 1 0 1 0 1 1 0 0 0"

python src/trainer.py \
    --output_dir="output" \
    --data_path="data/cifar10" \
    --network_type="cnn" \
    --search_for="macro" \
    --controller_training=FALSE \
    --cnn_macro_num_layers=12 \
    --batch_size=100 \
    --num_epochs=310 \
    --child_fixed_arc="${fixed_arc}" \
