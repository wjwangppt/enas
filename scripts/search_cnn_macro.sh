#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/trainer.py \
    --output_dir="output" \
    --data_path="data/cifar10" \
    --network_type="cnn" \
    --search_for="macro" \
    --controller_training=TRUE  \
    --cnn_macro_num_layers=12 \
    --batch_size=128 \
    --num_epochs=310 \
