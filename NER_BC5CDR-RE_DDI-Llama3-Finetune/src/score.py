import argparse
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str)
    return parser.parse_args()

def extract_entities_no_regex(text):
    entities = set()
    if not isinstance(text, str):
        return entities

    tag_start = '<span class="chemical">'
    tag_end = '</span>'

    # Split text by the opening tag
    parts = text.split(tag_start)

    # Skip the first part
    for part in parts[1:]:
        # Split by closing tag to isolate the entity
        if tag_end in part:
            entity = part.split(tag_end)[0]
            entities.add(entity.strip().lower())
            
    return entities

def compute_metrics(data):
    tp = 0
    fp = 0
    fn = 0

    for item in data:
        truth_set = extract_entities_no_regex(item['truth'])
        pred_set = extract_entities_no_regex(item['prediction'])

        # Calculate intersection and differences
        tp += len(truth_set.intersection(pred_set))
        fp += len(pred_set - truth_set)
        fn += len(truth_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def main():
    args = parse_args()
    
    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.json_file} not found.")
        sys.exit(1)

    precision, recall, f1 = compute_metrics(data)

    print(f"File: {args.json_file}")
    print("-" * 20)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("-" * 20)

if __name__ == "__main__":
    main()