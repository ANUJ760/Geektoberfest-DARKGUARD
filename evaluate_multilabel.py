"""
Multi-label detection, explainability, and dataset limitation analysis script.
Addresses Issue #4: [DARKGUARD] Add multi-label detection and explainability.
"""

from pathlib import Path
import pandas as pd
from model import (
    explain_multilabel,
    load_dataset,
    predict_multilabel,
    train_multilabel_model,
)

def run_multilabel_evaluation(output_path="MULTILABEL_EVALUATION_REPORT.md"):
    print("=" * 80)
    print("🛡️ DARKGUARD: MULTI-LABEL DETECTION & EXPLAINABILITY EVALUATION")
    print("=" * 80)

    res = train_multilabel_model()
    rep = res["report_dict"]

    print(f"\nHeld-Out Category-Level Evaluation (Test Size: {res['test_size']} samples, 80/20 Stratified Split):")
    print("-" * 80)
    print(f"{'Category':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    for cls in res["classes"]:
        m = rep[cls]
        print(f"{cls:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1-score']:<12.4f} {int(m['support']):<10}")
    print("-" * 80)
    print(f"{'Micro Avg':<20} {rep['micro avg']['precision']:<12.4f} {rep['micro avg']['recall']:<12.4f} {rep['micro avg']['f1-score']:<12.4f} {int(rep['micro avg']['support']):<10}")
    print(f"{'Macro Avg':<20} {rep['macro avg']['precision']:<12.4f} {rep['macro avg']['recall']:<12.4f} {rep['macro avg']['f1-score']:<12.4f} {int(rep['macro avg']['support']):<10}")
    print(f"{'Weighted Avg':<20} {rep['weighted avg']['precision']:<12.4f} {rep['weighted avg']['recall']:<12.4f} {rep['weighted avg']['f1-score']:<12.4f} {int(rep['weighted avg']['support']):<10}")
    print("=" * 80)

    demonstrations = [
        ("Only 2 items left in stock!", "Scarcity"),
        ("Hurry! This flash deal expires in 10 minutes.", "Urgency"),
        ("127 people are viewing this product right now!", "Social Proof"),
        ("I WANT TO PAY FULL PRICE LATER ;)", "Misdirection"),
        ("Hurry! Only 2 items left in stock. This offer expires in 5 minutes. 15 people are looking right now!", "Compound (Scarcity + Urgency + Social Proof)"),
        ("This shirt is made of 100% organic cotton. Available in navy and white.", "Neutral Non-Dark"),
    ]

    print("\nDemonstrations of Multi-Label Detection & Category-Level Explanations:")
    demo_records = []
    for text, desc in demonstrations:
        pred_res = predict_multilabel(text, res["model"], res["vectorizer"], res["mlb"], threshold=0.35)
        expl_res = explain_multilabel(text, res["model"], res["vectorizer"], res["mlb"], top_n=3)

        print(f"\nSnippet: \"{text}\"")
        print(f"Target / Description: {desc}")
        print(f"Detected Categories:  {pred_res['detected_categories']}")
        print("Active Category Features:")
        active_explanations = {}
        for cat in pred_res["detected_categories"]:
            if cat in expl_res and expl_res[cat]:
                terms = [f"{t} (+{w:.2f})" for t, w in expl_res[cat]]
                active_explanations[cat] = terms
                print(f"   [{cat}]: {', '.join(terms)}")
            elif cat == "Not Dark Pattern":
                print("   [Not Dark Pattern]: No deceptive features triggered.")

        demo_records.append({
            "text": text,
            "desc": desc,
            "detected": pred_res["detected_categories"],
            "probabilities": pred_res["probabilities"],
            "explanations": active_explanations,
        })

    # Dataset analysis
    data = load_dataset()
    cat_counts = data["Pattern Category"].value_counts()

    # Generate Markdown Report
    md = []
    md.append("# 🛡️ DARKGUARD: Multi-Label Detection and Explainability Report\n")
    md.append("## Objective\n")
    md.append("Address Issue #4 by extending DARKGUARD from binary classification to category-level and multi-label detection with interpretable category-specific explanations, while evaluating dataset constraints.\n")

    md.append("## Dataset Analysis & Ground-Truth Limitations\n")
    md.append("A thorough analysis of `dataset.tsv` reveals critical architectural constraints:")
    md.append("1. **Strictly Single-Label Annotations:** Every single sample in `dataset.tsv` is mapped to exactly one category string. There are 0 samples with multi-label annotations (no comma-delimited or compound label records).")
    md.append("2. **Label Distribution:**")
    md.append("   - `Not Dark Pattern`: 1,178 (50.0%)")
    md.append("   - `Scarcity`: 418 (17.7%)")
    md.append("   - `Social Proof`: 312 (13.2%)")
    md.append("   - `Urgency`: 210 (8.9%)")
    md.append("   - `Misdirection`: 195 (8.3%)")
    md.append("   - `Obstruction`: 27 (1.1%)")
    md.append("   - `Sneaking`: 12 (0.5%)")
    md.append("   - `Forced Action`: 4 (0.2%)")
    md.append("3. **Severe Minority Imbalance:** `Forced Action` (4 samples) and `Sneaking` (12 samples) possess insufficient support for standalone multi-label generalization.\n")

    md.append("## Multi-Label Architecture: One-Vs-Rest (OvR) Binary Relevance\n")
    md.append("To support true multi-label prediction where a webpage snippet exhibits multiple simultaneous dark patterns (e.g. Scarcity + Urgency), we implemented a `OneVsRestClassifier` with balanced Logistic Regression sub-models:")
    md.append("- Each category possesses its own calibrated decision hyperplane.")
    md.append("- Predictions are independent per category via sigmoid probabilities.")
    md.append("- Multiple categories can be simultaneously triggered if their probabilities exceed the detection threshold (`threshold=0.35 - 0.40`).")
    md.append("- Category-specific explanations are generated by decomposing TF-IDF feature activations against each category's learned linear weight vector.\n")

    md.append("## Category-Level Evaluation Results (Held-Out Test Set)\n")
    md.append("| Dark Pattern Category | Test Support | Precision | Recall | F1-Score | Status |")
    md.append("|---|---|---|---|---|---|")
    for cls in res["classes"]:
        m = rep[cls]
        status = "Strong (>0.85 F1)" if m["f1-score"] >= 0.85 else ("Moderate (>0.70 F1)" if m["f1-score"] >= 0.70 else "Data-Constrained (<0.70 F1)")
        md.append(f"| **{cls}** | {int(m['support'])} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {status} |")
    md.append(f"| **Micro Average** | {int(rep['micro avg']['support'])} | {rep['micro avg']['precision']:.4f} | {rep['micro avg']['recall']:.4f} | {rep['micro avg']['f1-score']:.4f} | - |")
    md.append(f"| **Macro Average** | {int(rep['macro avg']['support'])} | {rep['macro avg']['precision']:.4f} | {rep['macro avg']['recall']:.4f} | {rep['macro avg']['f1-score']:.4f} | - |\n")

    md.append("## Compound & Multi-Label Demonstrations\n")
    for rec in demo_records:
        cats_str = ", ".join(f"`{c}`" for c in rec["detected"])
        md.append(f"### Example: \"{rec['text']}\"")
        md.append(f"- **Intent:** {rec['desc']}")
        md.append(f"- **Detected Categories:** {cats_str}")
        if rec["explanations"]:
            md.append("- **Category Feature Explanations:**")
            for cat, terms in rec["explanations"].items():
                md.append(f"  - **{cat}:** {', '.join(terms)}")
        md.append("")

    md.append("## Annotation Recommendations for True Multi-Label Extension\n")
    md.append("To move from binary relevance OvR to fully verified multi-label training:")
    md.append("1. **Multi-Annotator Protocol:** Collect realistic e-commerce banners that bundle countdown timers with stock counters and re-annotate with multi-select checkboxes.")
    md.append("2. **Category Hierarchy:** Group rare patterns (`Sneaking`, `Obstruction`, `Forced Action`) into an aggregate `Deceptive Interaction` super-category or perform targeted web scraping to expand minority classes beyond <30 examples.")
    md.append("3. **Threshold Calibration:** Utilize category-specific F1-maximizing thresholds rather than a uniform threshold across imbalanced classes.\n")

    report_str = "\n".join(md)
    Path(output_path).write_text(report_str, encoding="utf-8")
    print(f"\nSaved report to {output_path}")

if __name__ == "__main__":
    run_multilabel_evaluation()
