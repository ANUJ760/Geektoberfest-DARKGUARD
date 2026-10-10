# 🛡️ DARKGUARD

**AI-powered Dark Pattern Detector**

DARKGUARD is a Streamlit machine-learning prototype that analyzes website text and predicts whether it contains a dark pattern.

## Live Demo

**[Launch DARKGUARD](https://darkguard-m9approgfbavpoma8h9ebg5.streamlit.app/)**

## Problem

Online platforms sometimes use wording that creates artificial pressure or nudges users toward a decision. Examples include fake urgency, scarcity, and social proof.

## What the prototype does

A user pastes website text and DARKGUARD returns:

- Dark Pattern / Not Dark Pattern
- Model confidence
- TF-IDF features that influenced the prediction
- A separate quick scan for urgency, scarcity, and social-proof signals
- Basic model and dataset insights

## ML Pipeline

```
Website Text
    ↓
Basic Text Cleaning
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
Dark Pattern / Not Dark Pattern
```

## Baseline

The current baseline uses:

- 80/20 stratified train/test split
- TF-IDF with 1- and 2-word features
- Logistic Regression

The prototype achieved approximately **93.4% test accuracy** on the current dataset under this split.

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- GitHub

## Dataset

The dataset contains website snippets with a binary label and a pattern category.

Dataset source: [Yamanalab/ec-darkpattern](https://github.com/yamanalab/ec-darkpattern)

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Competition Challenge

DARKGUARD provides a working baseline rather than an empty starter project. Participants improve the model or its supporting NLP components through the open GitHub Issues.

### Open Challenges

- **Easy:** [Error analysis and evaluation](https://github.com/radhikadhingra820-hue/DARKGUARD/issues/1)
- **Medium:** [Improve short and subtle text detection](https://github.com/radhikadhingra820-hue/DARKGUARD/issues/2)
- **Medium:** [Compare classifiers and improve detection](https://github.com/radhikadhingra820-hue/DARKGUARD/issues/3)
- **Hard:** [Add multi-label detection and explanations](https://github.com/radhikadhingra820-hue/DARKGUARD/issues/4)

For ML changes, contributors should use held-out evaluation data and report accuracy, precision, recall, F1-score, and a short explanation of the approach. Difficulty labels describe the expected scope, not a guaranteed time to complete.

## Project Structure

```
DARKGUARD/
├── app.py
├── model.py
├── dataset.tsv
├── requirements.txt
├── CONTRIBUTING.md
├── PRESENTATION.md
├── LICENSE
├── .gitignore
└── tests/
    └── test_model.py
```

## Notes

- The deployed app runs the baseline model from the `main` branch.
- A model prediction is an indicator produced from the training data; it is not proof of intent, deception, or harm by a website.
- Competition tasks and improvement work are kept in GitHub Issues so the prototype stays focused on the working baseline.
