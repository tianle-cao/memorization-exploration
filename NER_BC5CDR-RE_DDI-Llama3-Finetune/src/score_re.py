import argparse
import json
import sys
import re
from sklearn.metrics import f1_score, accuracy_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str)
    return parser.parse_args()

def clean_text(text):
    return str(text).strip().lower()

def extract_strict_label(text):
    text = clean_text(text)
    
    pattern = r'(ddi-false|ddi-effect|ddi-mechanism|ddi-advise|ddi-int)'
    matches = re.findall(pattern, text)
    
    if matches:
        return matches[0]
    else:
        return 'unknown'

def main():
    args = parse_args()
    
    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.json_file} not found.")
        sys.exit(1)

    y_true = []
    y_pred = []
    
    labels = ['ddi-false', 'ddi-effect', 'ddi-mechanism', 'ddi-advise', 'ddi-int']

    for item in data:
        truth = clean_text(item['truth'])
        

        if truth not in labels:
            match = re.search(r'(ddi-\w+)', truth)
            if match:
                truth = match.group(1)
        
        pred = extract_strict_label(item['prediction'])
    
        
        y_true.append(truth)
        y_pred.append(pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=labels)

    print(f"\nEvaluating File: {args.json_file}")
    print("=" * 60)
    print(f"Overall Accuracy : {accuracy:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Weighted F1      : {weighted_f1:.4f}")
    print("=" * 60)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))
    print("=" * 60)

if __name__ == "__main__":
    main()