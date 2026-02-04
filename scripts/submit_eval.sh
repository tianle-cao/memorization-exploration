#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00 
#SBATCH --output=logs/eval_%j.out

module load miniconda/24.3.0 2>/dev/null
source activate mimic_env 2>/dev/null || conda activate mimic_env

# fix
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

BASE_MODEL="model_cache/llama3.1_8b_instruct"
ADAPTER_PATH="checkpoints/llama3.1-tldr-v1" 
DATA_PATH="data/train.jsonl"

echo "Evaluating test set..."

python src/run_eval.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --data_path "$DATA_PATH"

echo "Evaluation finished."