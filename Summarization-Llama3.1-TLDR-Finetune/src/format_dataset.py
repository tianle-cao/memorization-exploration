import argparse
import os
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

# Configuration
DEFAULT_DATASET = "CarperAI/openai_summarize_tldr"
SYSTEM_PROMPT = "You are a helpful AI assistant specialized in summarizing text. Please provide a concise summary of the following content."

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Local path or HF ID")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET, help="Local path or HF ID")
    parser.add_argument("--num_proc", type=int, default=16, help="CPU cores for parallel processing")
    return parser.parse_args()

def format_batch(batch, tokenizer):
    texts = []
    
    # Iterate through batch
    for content, summary in zip(batch['content'], batch['summary']):
        if not content or not summary:
            texts.append("")
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
            {"role": "assistant", "content": summary}
        ]
        
        # apply_chat_template handles special tokens (BOS, EOS, headers)
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        
    return {"text": texts}

def main():
    args = parse_args()
    
    # Load Tokenizer
    print(f"Loading Tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load Dataset
    print(f"Loading dataset from {args.dataset_path}...")
    if os.path.isdir(args.dataset_path):
        ds = load_from_disk(args.dataset_path)
    else:
        # Load specific split (usually 'train')
        ds = load_dataset(args.dataset_path, split="train")

    # Standardize Column Names
    if "prompt" in ds.column_names and "label" in ds.column_names:
        print("Mapping dataset columns: prompt -> content, label -> summary")
        ds = ds.rename_columns({"prompt": "content", "label": "summary"})

    print(f"Processing {len(ds)} samples using {args.num_proc} cores...")

    # Filter empty data
    ds = ds.filter(
        lambda x: x['content'] is not None and x['summary'] is not None, 
        num_proc=args.num_proc
    )

    # Apply formatting
    ds = ds.map(
        format_batch,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        desc="Formatting"
    )

    # Remove failed formatting rows
    ds = ds.filter(lambda x: len(x['text']) > 0, num_proc=args.num_proc)

    # Save to JSONL
    print(f"Saving to {args.save_path}...")
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(args.save_path, force_ascii=False, num_proc=args.num_proc)
    
    print("Done.")

if __name__ == "__main__":
    main()