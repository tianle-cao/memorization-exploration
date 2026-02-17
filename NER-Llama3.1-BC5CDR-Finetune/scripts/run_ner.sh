#!/bin/bash
#SBATCH --job-name=NER-Llama3    
#SBATCH --partition=gpu          
#SBATCH --gres=gpu:a100:1           
#SBATCH --cpus-per-task=8        
#SBATCH --mem=32G                
#SBATCH --time=24:00:00          
#SBATCH --output=logs/run_ner_%j.out

module load miniconda/24.3.0 2>/dev/null
source activate mimic_env 2>/dev/null || conda activate mimic_env

MODEL_PATH="model_cache/llama3.1_8b_instruct"

DATASETS=('BC5CDR') 


for dataset in "${DATASETS[@]}"; do

    OUTPUT_DIR="checkpoints/ner-${dataset}-v1"
    DATA_DIR="data"

    python src/train.py \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR"

    echo "Finished processing $dataset"
done
