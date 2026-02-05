#!/bin/bash
#SBATCH --job-name=eval-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00 
#SBATCH --output=logs/eval_base_%j.out

module load miniconda/24.3.0 2>/dev/null

source activate mimic_env 2>/dev/null || conda activate mimic_env

# fix
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

BASE_MODEL="model_cache/llama3.1_8b_instruct"
DATA_PATH="data/train.jsonl"

echo "Evaluating BASELINE..."

python src/eval_baseline.py \
--base_model "$BASE_MODEL" \
--data_path "$DATA_PATH"

echo "Baseline Evaluation finished."
