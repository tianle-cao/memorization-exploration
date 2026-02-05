import argparse
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load Metric
    try:
        rouge = evaluate.load("rouge")
    except Exception:
        import os
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        rouge = evaluate.load("rouge")

    # Load Model & Tokenizer
    print(f">> Loading Base Model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    
    print(f">> Loading Adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # Load Test Data
    print(">> Loading Dataset...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    test_dataset = dataset.train_test_split(test_size=0.05, seed=42)["test"]
    
    print(f">> Starting Evaluation on {len(test_dataset)} samples...")

    predictions = []
    references = []

    # Inference Loop
    for i, item in enumerate(tqdm(test_dataset, desc="Generating")):

        split_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        if split_token in item["text"]:
            parts = item["text"].split(split_token)
            
            prompt = parts[0] + split_token
            
            ground_truth = parts[1].replace("<|eot_id|>", "").strip()
        else:
                continue

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        predictions.append(response)
        references.append(ground_truth)

        if i < 3:
            print(f"\n[Example {i+1}]")
            print(f"GT:   {ground_truth[:100]}...")
            print(f"PRED: {response[:100]}...")

    # Compute & Print Metrics
    print(">> Calculating ROUGE Scores...")

    results = rouge.compute(predictions=predictions, references=references, use_aggregator=True)
    
    print(f"ROUGE-1: {results['rouge1'] * 100:.2f}")
    print(f"ROUGE-2: {results['rouge2'] * 100:.2f}")
    print(f"ROUGE-L: {results['rougeL'] * 100:.2f}")


if __name__ == "__main__":
    main()