# NER_BC5CDR-RE_DDI-Llama3-Finetune

This project fine-tunes `Meta-Llama-3.1-8B-Instruct` on two core biomedical NLP tasks: 

1. Named Entity Recognition (NER) using the `BC5CDR` dataset for extracting *chemical* entities.

2. Relation Extraction (RE) using the `DDI 2013` dataset for classifying Drug-Drug Interactions.



## Results



### Task 1: Named Entity Recognition (NER) on BC5CDR-Chemical

Evaluation is strictly based on Entity-Level Exact Match.



| Model Status | Precision (%) | Recall (%) | Entity F1 (%) |
| :--- | :---: | :---: | :---: |
| Before Finetuning | 37.75 | 76.66 | 50.59 |
| After Finetuning | 92.39 | 90.82 | **91.60** |



### Task 2: Relation Extraction (RE) on DDI 2013

Evaluation is strictly based on standard classification parsing across 5 imbalanced categories (`DDI-false`, `DDI-effect`, `DDI-mechanism`, `DDI-advise`, `DDI-int`). 



| Model Status | Macro F1 (%) |
| :--- | :---: |
| Before Finetuning | 6.62 |
| After Finetuning | **79.91** |



## Acknowledgement
`train.py` is adapted from: [BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/multiple_GPU_train_V2-NER.py](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/blob/v1.0.0/llama/src/NER/multiple_GPU_train_V2-NER.py).