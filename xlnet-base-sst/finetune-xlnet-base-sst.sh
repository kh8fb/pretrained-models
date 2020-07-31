#!/bin/bash

# Script for running distilbert pad-baseline attributions from a master sbatch script
set -Eeuo pipefail
set -x

PYTHON="$HOME/anaconda3/envs/bert-itpt/bin/python3.8"
SCRIPT="/scratch/kh8fb/2020summer_research/pretrained-models/xlnet-base-sst/train_xlnet.py"
DATA_DIR="/scratch/kh8fb/2020summer_research/trained_models/xlnet_sst/stanfordSentimentTreebank/"
OUTPUT_DIR="/scratch/kh8fb/2020summer_research/trained_models/xlnet_sst/finetuned_xlnet_base_sst_more_epochs"

"$PYTHON" "$SCRIPT" \
    --data_dir "$DATA_DIR" \
    --task_name sst \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 12 \
    --max_seq_length 512 \
