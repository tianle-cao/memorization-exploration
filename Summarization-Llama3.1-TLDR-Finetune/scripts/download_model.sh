#!/bin/bash

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
SAVE_DIR="model_cache/llama3.1_8b_instruct"

echo ">> Checking environment..."
source activate mimic_env 2>/dev/null || conda activate mimic_env
echo ">> Starting download for: $MODEL_ID"
echo ">> Save location: $(pwd)/$SAVE_DIR"


python -c "
from huggingface_hub import snapshot_download
import sys

try:
    snapshot_download(
        repo_id='$MODEL_ID',
        local_dir='$SAVE_DIR',
        local_dir_use_symlinks=False,
        ignore_patterns=['*.msgpack', '*.h5', '*.ot', '*.onnx'] # Ignore non-PyTorch files
    )
    print('Download Success!')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"