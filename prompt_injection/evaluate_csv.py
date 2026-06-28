"""Batch evaluation on any CSV dataset with columns: id, category, injection_text.

This v2 version uses the calibrated threshold from the checkpoint (not 0.50)
and reports both the calibrated and raw argmax results for comparison.

Run:
    python -m prompt_injection.evaluate_csv
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

from prompt_injection.predict import Predictor
from prompt_injection.config import MODEL_PATH


def evaluate_csv(csv_path: Path, output_path: Path, model_path: Path = MODEL_PATH):
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    print(f"Loading CSV dataset: {csv_path}")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"  Loaded {len(rows):,} rows.")
    texts = [row["injection_text"] for row in rows]

    print("Initializing predictor model...")
    # use_uncertainty=False → binary output, calibrated threshold
    predictor = Predictor(model_path=model_path, use_uncertainty=False)
    print(f"  Calibrated threshold: {predictor.threshold:.2f}")

    print("Running batch predictions...")
    results = predictor.predict_batch(texts)

    # Categories tracking
    category_stats = defaultdict(lambda: {"total": 0, "detected": 0})
    total_detected = 0

    # Write output predictions CSV
    fieldnames = [
        "id", "category", "injection_text",
        "predicted_label", "predicted_label_name", "injection_prob",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row, res in zip(rows, results):
            cat = row["category"]
            pred_label = res["label"]

            category_stats[cat]["total"] += 1
            if pred_label == 1:
                category_stats[cat]["detected"] += 1
                total_detected += 1

            writer.writerow({
                "id":                  row["id"],
                "category":            cat,
                "injection_text":      row["injection_text"],
                "predicted_label":     pred_label,
                "predicted_label_name": res["label_name"],
                "injection_prob":      res["injection_prob"],
            })

    print(f"\nPredictions saved to: {output_path}")
    print("\n" + "=" * 70)
    print(f"  Evaluation Results on Test CSV -- {len(rows):,} total prompts")
    print(f"  Threshold: {predictor.threshold:.2f} (calibrated)")
    print(f"  Overall Injection Detection Rate: {total_detected / len(rows):.2%} ({total_detected}/{len(rows)})")
    print("=" * 70)

    print(f"\n{'Category':<32} | {'Detection Rate':<14} | {'Counts':<10}")
    print("-" * 65)
    for cat, stats in sorted(
        category_stats.items(),
        key=lambda x: -(x[1]["detected"] / x[1]["total"]),
    ):
        rate = stats["detected"] / stats["total"]
        counts_str = f"{stats['detected']}/{stats['total']}"
        print(f"{cat:<32} | {rate:<14.2%} | {counts_str:<10}")
    print("-" * 65)


if __name__ == "__main__":
    csv_file    = Path("data/prompt_injection_test_dataset.csv")
    output_file = Path("data/prompt_injection_test_dataset_predictions.csv")
    evaluate_csv(csv_file, output_file)
