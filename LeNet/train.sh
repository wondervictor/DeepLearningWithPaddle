#!/usr/bin/env bash

set -e


trainer=LeNet.py

paddle train \
    --config=$trainer \
    --save_dir=output \
    --trainer_count=2 \
    --log_period=1000 \
    --dot_period=100 \
    --num_passes=50 \
    --use_gpu=false \
    --show_parameter_stats_period=3000 \
    2>&1 | tee train.log

