import argparse
import json
import torch
import os
from tqdm import tqdm
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Llama 3 (Base or Adapter).")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the fine-tuned LoRA adapter (Optional)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test dataset (CSV)")
    parser.add_argument("--output_file", type=str, default="results/predictions.json", help="Output JSON file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load Base Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )

    # Load LoRA adapter
    if args.adapter_path:
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print("No adapter path provided. Running in Base Model mode.")

    model.eval()

    # Load Test Data
    context_feat = Features({'unprocessed': Value('string'), 'processed': Value('string')})
    dataset = load_dataset("csv", data_files=args.data_path, split="train", features=context_feat)

    def collate_fn(batch):
        return {
            'prompts': [item['unprocessed'] for item in batch],
            'truths': [item['processed'] for item in batch]
        }

    # DataLoader to run parallelly
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    results = []


    for batch in tqdm(dataloader):
        prompts = batch['prompts']
        truths = batch['truths']
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for prompt, pred, truth in zip(prompts, decoded_preds, truths):
            results.append({
                "input": prompt,
                "truth": truth,
                "prediction": pred.strip()
            })

    # Save Results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()