# adapted from: https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/blob/v1.0.0/llama/src/NER/multiple_GPU_train_V2-NER.py

from datasets import load_dataset, Features, Value, disable_caching
import torch, os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from utils import find_all_linear_names, print_trainable_parameters
from accelerate import Accelerator

accelerator = Accelerator()
device_index = Accelerator().process_index
device_map = {"": device_index}
import argparse

disable_caching()

parser = argparse.ArgumentParser(description='Process command-line arguments.')
# MODIFIED: Updated arguments to match your new folder structure and variable names
parser.add_argument('--model_path', type=str, required=True, help='Path to local model weights')
parser.add_argument('--data_dir', type=str, required=True, help='Path to data folder')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

args = parser.parse_args()

# MODIFIED: Use passed arguments
model_name = args.model_path
data_dir = args.data_dir
output_dir = args.output_dir

context_feat = Features({'unprocessed': Value(dtype='string', id=None),'processed': Value(dtype='string', id=None)})

#NER
# MODIFIED: Construct paths dynamically based on data_dir argument
train_file = os.path.join(data_dir, "sentence_level_train.csv")
dev_file = os.path.join(data_dir, "sentence_level_dev.csv")

train_dataset = load_dataset("csv", data_files=[train_file], split="train", features=context_feat)
eval_dataset = load_dataset("csv", data_files=[dev_file], split="train", features=context_feat)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# MODIFIED: Added local_files_only=True for HPC environment safety
base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                  torch_dtype=torch.bfloat16, 
                                                  quantization_config=bnb_config,
                                                  device_map=device_map,
                                                  local_files_only=True)


base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

# MODIFIED: Load tokenizer from local model_path (Llama 3), removed hardcoded Llama 2 path
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
# MODIFIED: Fix for Llama 3 which has no default pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    #target_modules=find_all_linear_names(base_model),
    target_modules= [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "gate_proj",
        "up_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)
#base_model = accelerator.prepare(base_model)

def formatting_prompts_func(example):
    # OPTIMIZED: Removed loop to handle single example input correctly
    return f"{example['unprocessed']} {example['processed']}"


# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = SFTConfig( # MODIFIED: Change to SFTConfig because of trl version
    output_dir=output_dir,
    max_length=1500, # MODIFIED: Move to here because of trl version
    # OPTIMIZED: Increased batch size and grad accumulation for Llama 3 efficiency
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing = True,
    max_grad_norm= 0.3,
    num_train_epochs=3, 
    # OPTIMIZED: Adjusted learning rate for QLoRA
    learning_rate=2e-4,
    bf16=True,
    save_strategy="epoch",
    save_total_limit=10,
    logging_steps=10,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    weight_decay=0.00001,
    warmup_ratio=0.01,
    ddp_find_unused_parameters=False,
    eval_strategy="epoch",
    report_to="none", # MODIFIED: clean logs
    #load_best_model_at_end=True,
    #metric_for_best_model='eval_loss'
)


trainer = SFTTrainer(
    base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # MODIFIED: Added eval_dataset to Trainer
    processing_class=tokenizer,
    #max_seq_length=1500, # MODIFIED: Move to SFTConfig because of trl version
    formatting_func=formatting_prompts_func,
    args=training_args,
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

#trainer.train(resume_from_checkpoint=True) 
trainer.train() 
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)