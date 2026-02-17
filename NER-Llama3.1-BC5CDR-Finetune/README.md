# NER-Llama3.1-BC5CDR-Finetune

This project finetunes `Meta-Llama-3.1-8B-Instruct` on the `BC5CDR` dataset to improve **Named Entity Recognition (NER)** capabilities for chemical and disease entities.

## Results

Comparison of Precision, Recall, and F1 scores before and after finetuning:

| Model Status | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Before Finetuning** | 37.75 | 76.66 | 50.59 |
| **After Finetuning** | **92.39** | **90.82** | **91.60** |

## Acknowledgement
`train.py` is adapted from: [BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/multiple_GPU_train_V2-NER.py](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/blob/v1.0.0/llama/src/NER/multiple_GPU_train_V2-NER.py).
