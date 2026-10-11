import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).resolve().parent / "dataset.tsv"

# Stop words to retain because they convey key urgency, scarcity, and immediacy cues
SIGNAL_WORDS = {
    "only", "now", "will", "out", "all", "more", "most", "few",
    "less", "over", "under", "here", "last", "no", "not", "off",
}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - SIGNAL_WORDS)

def build_vectorizer(use_enhanced=True, char_min_df=2):
    if not use_enhanced:
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                stop_words=CUSTOM_STOP_WORDS,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=char_min_df,
                sublinear_tf=True,
            ),
        ),
    ])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_dataset():
    data = pd.read_csv(DATASET_PATH, sep="\t")

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError(
            "dataset.tsv must contain 'text' and 'label' columns."
        )

    data["text"] = data["text"].fillna("")
    data["clean_text"] = data["text"].apply(clean_text)

    return data

def train_model(use_enhanced=True, C=1.5):
    data = load_dataset()

    X = data["clean_text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer = build_vectorizer(use_enhanced=use_enhanced)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=C)
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Not Dark Pattern", "Dark Pattern"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "model": model,
        "vectorizer": vectorizer,
        "data": data,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "report": report,
        "feature_count": len(vectorizer.get_feature_names_out()),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

def predict_text(text, model, vectorizer):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])

    prediction = int(model.predict(text_tfidf)[0])
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = float(probabilities[prediction])

    return prediction, confidence, probabilities

def explain_prediction(text, model, vectorizer, top_n=6):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()
    contributions = text_tfidf.toarray()[0] * model.coef_[0]

    cleaned_names = [f.split("__", 1)[-1].strip() for f in feature_names]

    positive = contributions.argsort()[::-1]
    positive_terms = [
        (cleaned_names[i], float(contributions[i]))
        for i in positive
        if contributions[i] > 0
    ][:top_n]

    negative = contributions.argsort()
    negative_terms = [
        (cleaned_names[i], float(contributions[i]))
        for i in negative
        if contributions[i] < 0
    ][:top_n]

    return positive_terms, negative_terms
