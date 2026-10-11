import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

DATASET_PATH = Path(__file__).resolve().parent / "dataset.tsv"

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

def train_model():
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

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
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

    positive = contributions.argsort()[::-1]
    positive_terms = [
        (feature_names[i], float(contributions[i]))
        for i in positive
        if contributions[i] > 0
    ][:top_n]

    negative = contributions.argsort()
    negative_terms = [
        (feature_names[i], float(contributions[i]))
        for i in negative
        if contributions[i] < 0
    ][:top_n]

    return positive_terms, negative_terms

def train_multilabel_model(data=None, test_size=0.20, random_state=42):
    if data is None:
        data = load_dataset()

    data_copy = data.copy()
    data_copy["category_list"] = data_copy["Pattern Category"].apply(lambda c: [c])

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(data_copy["category_list"])

    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        data_copy["clean_text"],
        Y,
        data_copy.index,
        test_size=test_size,
        random_state=random_state,
        stratify=data_copy["Pattern Category"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    )
    model.fit(X_train_tfidf, Y_train)

    predictions = model.predict(X_test_tfidf)

    report_dict = classification_report(
        Y_test,
        predictions,
        target_names=mlb.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        Y_test,
        predictions,
        target_names=mlb.classes_,
        zero_division=0,
    )

    return {
        "model": model,
        "vectorizer": vectorizer,
        "mlb": mlb,
        "classes": list(mlb.classes_),
        "data": data_copy,
        "report_dict": report_dict,
        "report_text": report_text,
        "feature_count": len(vectorizer.get_feature_names_out()),
        "train_size": len(idx_train),
        "test_size": len(idx_test),
        "X_test": X_test,
        "Y_test": Y_test,
        "predictions": predictions,
    }

def predict_multilabel(text, model, vectorizer, mlb, threshold=0.40):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(text_tfidf)[0]
    prob_dict = {cls: float(prob) for cls, prob in zip(mlb.classes_, probabilities)}

    dark_categories = [cls for cls in mlb.classes_ if cls != "Not Dark Pattern"]
    detected_dark = [cls for cls in dark_categories if prob_dict[cls] >= threshold]

    if detected_dark:
        detected = detected_dark
        is_dark = True
    else:
        is_dark = prob_dict.get("Not Dark Pattern", 0.0) < 0.5 and any(prob_dict[c] >= 0.25 for c in dark_categories)
        if is_dark:
            top_category = max(dark_categories, key=lambda c: prob_dict[c])
            detected = [top_category]
        else:
            detected = ["Not Dark Pattern"]

    return {
        "detected_categories": detected,
        "is_dark": is_dark,
        "probabilities": prob_dict,
    }

def explain_multilabel(text, model, vectorizer, mlb, top_n=4):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    text_array = text_tfidf.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()

    explanations = {}
    for i, estimator in enumerate(model.estimators_):
        cat_name = mlb.classes_[i]
        coef = estimator.coef_[0]
        contributions = text_array * coef

        pos_indices = contributions.argsort()[::-1]
        top_terms = [
            (feature_names[j], float(contributions[j]))
            for j in pos_indices
            if contributions[j] > 0
        ][:top_n]

        explanations[cat_name] = top_terms

    return explanations

