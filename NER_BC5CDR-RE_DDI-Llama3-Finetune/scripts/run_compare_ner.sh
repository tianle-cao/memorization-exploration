#!/bin/bash
#SBATCH --job-name=Compare-NER
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/compare_%j.out

# Load Environment
module load miniconda/24.3.0 2>/dev/null
source activate mimic_env 2>/dev/null || conda activate mimic_env

# Config
BASE_MODEL="model_cache/llama3.1_8b_instruct"
ADAPTER_PATH="checkpoints/ner-BC5CDR-v1/final_checkpoint"
DATA_PATH="data/sentence_level_test.csv"

# Outputs
OUT_BASE="results/pred_before.json"
OUT_FT="results/pred_after.json"


echo "1: Baseline Evaluation"
python src/inference.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_file "$OUT_BASE"

echo "Calculating Baseline Score..."
python src/score.py "$OUT_BASE"


echo "2: Finetuned Evaluation"
python src/inference.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --data_path "$DATA_PATH" \
    --output_file "$OUT_FT"

echo "Calculating Finetuned Score..."
python src/score.py "$OUT_FT"

echo "Comparison Done."