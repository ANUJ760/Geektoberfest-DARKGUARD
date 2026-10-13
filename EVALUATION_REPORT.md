# 🛡️ DARKGUARD Error Analysis and Evaluation Report

## Executive Summary

This report documents a systematic evaluation and error analysis of the baseline DARKGUARD model on a held-out test set.

- **Model:** TF-IDF (1-2 ngrams, English stopwords) + Logistic Regression (`max_iter=1000`)
- **Data Split:** 80/20 Stratified Split (`random_state=42`)
- **Train Samples:** 1884
- **Test Samples:** 472
- **Accuracy:** 93.43%
- **Precision (Dark Pattern):** 0.9638
- **Recall (Dark Pattern):** 0.9025
- **F1-Score (Dark Pattern):** 0.9322

## Confusion Matrix

| | Predicted: Not Dark | Predicted: Dark | Total |
|---|---|---|---|
| **Actual: Not Dark** | 228 (True Negatives) | 8 (False Positives) | 236 |
| **Actual: Dark** | 23 (False Negatives) | 213 (True Positives) | 236 |
| **Total** | 251 | 221 | 472 |

## Classification Report

```
precision    recall  f1-score   support

Not Dark Pattern       0.91      0.97      0.94       236
    Dark Pattern       0.96      0.90      0.93       236

        accuracy                           0.93       472
       macro avg       0.94      0.93      0.93       472
    weighted avg       0.94      0.93      0.93       472
```

## Category-Level Error Breakdown (Dark Patterns)

| Dark Pattern Category | Test Support | Correctly Identified | False Negatives | Recall |
|---|---|---|---|---|
| Misdirection | 33 | 28 | 5 | 84.85% |
| Social Proof | 62 | 58 | 4 | 93.55% |
| Sneaking | 6 | 2 | 4 | 33.33% |
| Urgency | 42 | 38 | 4 | 90.48% |
| Forced Action | 3 | 0 | 3 | 0.00% |
| Scarcity | 84 | 82 | 2 | 97.62% |
| Obstruction | 6 | 5 | 1 | 83.33% |

## Error Analysis & Representative Examples

### 1. False Positives (8 occurrences)
False positives occur when normal e-commerce language is misclassified as manipulative. Common triggers include standard stock statuses and transaction text:
- `" Frame out of stock in this colour"` (Estimated Dark probability: 88.70%)
- `" Take time to pay."` (Estimated Dark probability: 84.69%)
- `"Discontinued Item"` (Estimated Dark probability: 75.38%)
- `"Pay With:"` (Estimated Dark probability: 56.23%)
- `" ALL SALE "` (Estimated Dark probability: 52.27%)
- `"You have no items in your basket."` (Estimated Dark probability: 57.47%)

### 2. False Negatives (23 occurrences)
False negatives represent dark patterns that escaped detection (model predicted Not Dark Pattern). These are critical as manipulative language goes unnoticed:
- **[Social Proof]** `"Someone added 1pc 4ocean natural stone beads beaded bracelet men boho handmade to their cart RIGHT NOW"` (Estimated Dark probability: 39.97%)
- **[Misdirection]** `"I WANT TO PAY MORE LATER ;)"` (Estimated Dark probability: 47.19%)
- **[Sneaking]** `"WOULD YOU LIKE A FREE NEXT DIRECTORY? Explore all the latest collections from your free Next Directory. By opting in to receive the Next Directory you agree that we may send you, at our discretion, further Directories usually in January and July each year, at a charge of £3.75 per publication. You also agree to receive supplementary Directories and ‘New In’ publications at no additional charge."` (Estimated Dark probability: 38.59%)
- **[Social Proof]** `"Our most recent customers are from New Rochelle, Roseville, and Brooksville"` (Estimated Dark probability: 28.64%)
- **[Scarcity]** `"Will sell out fast!"` (Estimated Dark probability: 40.71%)
- **[Misdirection]** `"OUR BEST SELLING GOGGLE"` (Estimated Dark probability: 31.81%)
- **[Urgency]** `"HAUTELOOK 04:53"` (Estimated Dark probability: 43.15%)
- **[Forced Action]** `"I agree to receive marketing emails from Natural Life and agree to our Privacy Policy and terms of use."` (Estimated Dark probability: 20.80%)

## Key Findings & Recommended Next Steps

1. **False Negative Asymmetry:** False negatives (23) outnumber false positives (8) by nearly 3:1. The classifier errs on the side of caution with its default 0.5 decision threshold.
2. **Low-Frequency Category Vulnerability:** Minority categories such as `Forced Action` (0% recall) and `Sneaking` (33.33% recall) are severely under-detected.
3. **Two Most Valuable Weaknesses to Address:**
   - **Weakness 1:** Short, context-poor, or subtle text snippets (e.g. countdowns, 'Will sell out fast!') with sparse TF-IDF features
   - **Weakness 2:** Low-frequency dark pattern categories (Forced Action with 0% recall, Sneaking with 33% recall) due to class imbalance and decision threshold
