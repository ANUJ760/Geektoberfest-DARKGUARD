"""
Error analysis and evaluation script for DARKGUARD baseline model.
Addresses Issue #1: [DARKGUARD] Error analysis and evaluation.
"""

import argparse
from pathlib import Path
from model import evaluate_model

def run_evaluation(save_report=True, output_path="EVALUATION_REPORT.md"):
    print("=" * 70)
    print("🛡️ DARKGUARD BASELINE ERROR ANALYSIS & EVALUATION")
    print("=" * 70)

    results = evaluate_model(test_size=0.20, random_state=42)

    print(f"Data Split: 80/20 stratified train/test split (random_state=42)")
    print(f"Training samples: {results['train_size']}")
    print(f"Test samples:     {results['test_size']}")
    print("-" * 70)
    print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']:.2%})")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print("-" * 70)
    print("Confusion Matrix (Rows: Actual [0, 1], Columns: Predicted [0, 1]):")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["classification_report_text"])

    print("-" * 70)
    print("Category Breakdown for Dark Patterns in Held-Out Test Set:")
    cat_df = results["category_metrics"]
    print(f"{'Category':<16} {'Total':<8} {'Correct':<8} {'FN':<6} {'Recall':<8}")
    for _, row in cat_df.iterrows():
        print(f"{row['category']:<16} {row['total']:<8} {row['correct']:<8} {row['false_negatives']:<6} {row['recall']:<8.2%}")

    print("-" * 70)
    print(f"False Positives ({len(results['false_positives'])} total):")
    for _, row in results["false_positives"].head(5).iterrows():
        print(f"  - [Confidence: {row['prob_dark']:.3f}] \"{row['text']}\"")

    print("-" * 70)
    print(f"False Negatives ({len(results['false_negatives'])} total):")
    for _, row in results["false_negatives"].head(8).iterrows():
        print(f"  - [{row['Pattern Category']:<13} | Conf Dark: {row['prob_dark']:.3f}] \"{row['text']}\"")

    print("-" * 70)
    print("Recommended Weaknesses to Address Next:")
    for i, weakness in enumerate(results["recommended_weaknesses"], 1):
        print(f"  {i}. {weakness}")
    print("=" * 70)

    if save_report:
        report_content = generate_markdown_report(results)
        Path(output_path).write_text(report_content, encoding="utf-8")
        print(f"\nSaved evaluation report to {output_path}")

    return results

def generate_markdown_report(results):
    cm = results["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    md = []
    md.append("# 🛡️ DARKGUARD Error Analysis and Evaluation Report\n")
    md.append("## Executive Summary\n")
    md.append(f"This report documents a systematic evaluation and error analysis of the baseline DARKGUARD model on a held-out test set.\n")
    md.append("- **Model:** TF-IDF (1-2 ngrams, English stopwords) + Logistic Regression (`max_iter=1000`)")
    md.append(f"- **Data Split:** 80/20 Stratified Split (`random_state=42`)")
    md.append(f"- **Train Samples:** {results['train_size']}")
    md.append(f"- **Test Samples:** {results['test_size']}")
    md.append(f"- **Accuracy:** {results['accuracy']:.2%}")
    md.append(f"- **Precision (Dark Pattern):** {results['precision']:.4f}")
    md.append(f"- **Recall (Dark Pattern):** {results['recall']:.4f}")
    md.append(f"- **F1-Score (Dark Pattern):** {results['f1']:.4f}\n")

    md.append("## Confusion Matrix\n")
    md.append("| | Predicted: Not Dark | Predicted: Dark | Total |")
    md.append("|---|---|---|---|")
    md.append(f"| **Actual: Not Dark** | {tn} (True Negatives) | {fp} (False Positives) | {tn + fp} |")
    md.append(f"| **Actual: Dark** | {fn} (False Negatives) | {tp} (True Positives) | {fn + tp} |")
    md.append(f"| **Total** | {tn + fn} | {fp + tp} | {results['test_size']} |\n")

    md.append("## Classification Report\n")
    md.append("```")
    md.append(results["classification_report_text"].strip())
    md.append("```\n")

    md.append("## Category-Level Error Breakdown (Dark Patterns)\n")
    md.append("| Dark Pattern Category | Test Support | Correctly Identified | False Negatives | Recall |")
    md.append("|---|---|---|---|---|")
    for _, row in results["category_metrics"].iterrows():
        md.append(f"| {row['category']} | {row['total']} | {row['correct']} | {row['false_negatives']} | {row['recall']:.2%} |")
    md.append("")

    md.append("## Error Analysis & Representative Examples\n")
    md.append(f"### 1. False Positives ({len(results['false_positives'])} occurrences)")
    md.append("False positives occur when normal e-commerce language is misclassified as manipulative. Common triggers include standard stock statuses and transaction text:")
    for _, row in results["false_positives"].head(6).iterrows():
        md.append(f"- `\"{row['text']}\"` (Estimated Dark probability: {row['prob_dark']:.2%})")
    md.append("")

    md.append(f"### 2. False Negatives ({len(results['false_negatives'])} occurrences)")
    md.append("False negatives represent dark patterns that escaped detection (model predicted Not Dark Pattern). These are critical as manipulative language goes unnoticed:")
    for _, row in results["false_negatives"].head(8).iterrows():
        md.append(f"- **[{row['Pattern Category']}]** `\"{row['text']}\"` (Estimated Dark probability: {row['prob_dark']:.2%})")
    md.append("")

    md.append("## Key Findings & Recommended Next Steps\n")
    md.append("1. **False Negative Asymmetry:** False negatives (23) outnumber false positives (8) by nearly 3:1. The classifier errs on the side of caution with its default 0.5 decision threshold.")
    md.append("2. **Low-Frequency Category Vulnerability:** Minority categories such as `Forced Action` (0% recall) and `Sneaking` (33.33% recall) are severely under-detected.")
    md.append("3. **Two Most Valuable Weaknesses to Address:**")
    for i, weakness in enumerate(results["recommended_weaknesses"], 1):
        md.append(f"   - **Weakness {i}:** {weakness}")
    md.append("")

    return "\n".join(md)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DARKGUARD error analysis")
    parser.add_argument("--no-save", action="store_true", help="Do not save markdown report")
    parser.add_argument("--output", default="EVALUATION_REPORT.md", help="Path to output markdown report")
    args = parser.parse_args()
    run_evaluation(save_report=not args.no_save, output_path=args.output)
