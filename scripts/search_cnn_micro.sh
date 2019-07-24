#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/trainer.py \
    --output_dir="output" \
    --data_path="data/cifar10" \
    --network_type="cnn" \
    --search_for="micro" \
    --controller_training=TRUE \
    --cnn_micro_num_layers=6 \
    --cnn_micro_num_cells=5 \
    --batch_size=160 \
    --num_epochs=150 \