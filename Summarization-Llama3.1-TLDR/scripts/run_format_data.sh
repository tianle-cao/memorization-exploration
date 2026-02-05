#!/bin/bash

module load miniconda/24.3.0
source activate mimic_env

MODEL_PATH="model_cache/llama3.1_tokenizer"
OUTPUT_PATH="data/train.jsonl"
NUM_PROC=8

python src/format_dataset.py \
    --save_path $OUTPUT_PATH \
    --model_path $MODEL_PATH \
    --num_proc $NUM_PROC