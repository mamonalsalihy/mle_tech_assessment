import pandas as pd
import string
import re
import sys

# Utility functions for evaluation
def normalize_answer(s):
    # Lowercase and remove punctuation/articles/whitespace for fairer comparison
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    # Compute F1 score at token level
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_common = sum(min(prediction_tokens.count(w), ground_truth_tokens.count(w)) for w in common)
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        # If either is empty, f1 is 1 if they are both empty, 0 otherwise
        return 1.0 if prediction_tokens == ground_truth_tokens else 0.0
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate(predictions, ground_truths):
    assert len(predictions) == len(ground_truths)
    em_total = 0
    f1_total = 0
    for pred, gt in zip(predictions, ground_truths):
        em_total += exact_match_score(pred, gt)
        f1_total += f1_score(pred, gt)
    em = (em_total / len(predictions)) * 100
    f1 = (f1_total / len(predictions)) * 100
    return em, f1

if __name__ == "__main__":
    # Command line arguments (optional): 
    # python analysis_script.py inference_data.csv predictions.csv
    # Or you can hardcode file paths below.
    if len(sys.argv) != 3:
        print("Usage: python analysis_script.py <inference_data.csv> <predictions.csv>")
        sys.exit(1)

    inference_data_path = sys.argv[1]
    predictions_path = sys.argv[2]

    # Load inference data
    # The inference_data.csv should have columns: query, title, abstract, label (the ground-truth answer)
    df = pd.read_csv(inference_data_path)

    print(df.columns)
    # Ground truth answers
    ground_truths = df["label"].tolist()

    # Load predictions - assuming predictions.csv has a column "prediction" corresponding to each input in order
    pred_df = pd.read_csv(predictions_path)
    predictions = pred_df["label"].tolist()

    # Compute metrics
    em, f1 = evaluate(predictions, ground_truths)

    # Print the results
    print(f"Exact Match (EM): {em:.2f}%")
    print(f"F1 Score: {f1:.2f}%")

    # Depending on your analysis needs, you may also print distributional information, 
    # error cases, or store results in a file.
    print("\nExamples of predictions and ground truths:")
    for i in range(min(10, len(predictions))):  # Show up to 10 examples
        print(f"Example {i + 1}:")
        print(f"Query: {df['query'][i]}")
        print(f"Ground Truth: {ground_truths[i]}")
        print(f"Prediction: {predictions[i]}")
        print(f"EM: {exact_match_score(predictions[i], ground_truths[i])}")
        print(f"F1: {f1_score(predictions[i], ground_truths[i]):.2f}\n")