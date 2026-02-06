#!/bin/bash
#SBATCH --job-name=llama3-tldr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=logs/train_%j.out

module load miniconda/24.3.0 2>/dev/null
source activate mimic_env 2>/dev/null || conda activate mimic_env

echo "Job started on $(hostname) at $(date)"

# Hyperparameters
MODEL_PATH="model_cache/llama3.1_8b_instruct"
DATA_PATH="data/train.jsonl"
OUTPUT_DIR="checkpoints/llama3.1-tldr-v1"

BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=2e-4
EPOCHS=1
MAX_LEN=2048

# Run
python src/train.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --max_seq_len $MAX_LEN

echo "Job finished at $(date)"