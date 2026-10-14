import json
from pathlib import Path

from model import (
    DEFAULT_MULTILABEL_THRESHOLD,
    clean_text,
    evaluate_model,
    evaluate_on_multilabel_dataset,
    explain_multilabel,
    explain_prediction,
    load_dataset,
    predict_multilabel,
    predict_text,
    train_model,
    train_multilabel_model,
)


def test_load_dataset():
    data = load_dataset()
    assert len(data) > 0
    assert "clean_text" in data.columns
    assert "label" in data.columns


def test_clean_text():
    sample = "  Hurry!   Limited Offer   "
    cleaned = clean_text(sample)
    assert cleaned == "hurry! limited offer"


def test_train_model():
    results = train_model()
    assert "accuracy" in results
    assert 0.85 <= results["accuracy"] <= 1.0
    assert results["feature_count"] > 100


def test_prediction_returns_valid_class():
    results = train_model()
    model = results["model"]
    vectorizer = results["vectorizer"]

    prediction, confidence, probabilities = predict_text(
        "Only 1 left in stock!",
        model,
        vectorizer,
    )

    assert prediction in [0, 1]
    assert 0.0 <= confidence <= 1.0
    assert len(probabilities) == 2


def test_evaluate_model():
    eval_res = evaluate_model(test_size=0.20, random_state=42)
    assert 0.90 <= eval_res["accuracy"] <= 1.0
    assert 0.85 <= eval_res["precision"] <= 1.0
    assert 0.85 <= eval_res["recall"] <= 1.0
    assert 0.85 <= eval_res["f1"] <= 1.0
    assert eval_res["confusion_matrix"].shape == (2, 2)
    assert "category_metrics" in eval_res
    assert len(eval_res["category_metrics"]) > 0
    assert len(eval_res["false_positives"]) > 0
    assert len(eval_res["false_negatives"]) > 0
    assert len(eval_res["recommended_weaknesses"]) == 2


def test_evaluate_model_reproducibility():
    run1 = evaluate_model(test_size=0.20, random_state=42)
    run2 = evaluate_model(test_size=0.20, random_state=42)
    assert run1["accuracy"] == run2["accuracy"]
    assert (run1["confusion_matrix"] == run2["confusion_matrix"]).all()
    assert len(run1["false_positives"]) == len(run2["false_positives"])
    assert len(run1["false_negatives"]) == len(run2["false_negatives"])


def test_evaluate_model_custom_split():
    eval_res = evaluate_model(test_size=0.15, random_state=123)
    assert 0.85 <= eval_res["accuracy"] <= 1.0
    assert eval_res["test_size"] == int(len(eval_res["test_df"]))


def test_evaluate_model_with_prefit_model():
    trained = train_model()
    eval_res = evaluate_model(
        model=trained["model"],
        vectorizer=trained["vectorizer"],
        test_size=0.20,
        random_state=42,
    )
    assert 0.90 <= eval_res["accuracy"] <= 1.0
    assert eval_res["confusion_matrix"].shape == (2, 2)


def test_evaluate_cli_run(tmp_path):
    from evaluate import run_evaluation
    out_file = tmp_path / "test_report.md"
    results = run_evaluation(save_report=True, output_path=str(out_file))
    assert 0.90 <= results["accuracy"] <= 1.0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "DARKGUARD Error Analysis" in content
    assert "Confusion Matrix" in content


# ============================================================================
# Multi-Label Detection, Explainability, & Benchmark Evaluation Tests
# ============================================================================

def test_train_multilabel_model():
    res = train_multilabel_model()
    assert "model" in res
    assert "mlb" in res
    assert len(res["classes"]) == 8
    assert "Scarcity" in res["classes"]
    assert "Urgency" in res["classes"]
    assert "Social Proof" in res["classes"]
    assert "report_dict" in res
    assert res["threshold"] == DEFAULT_MULTILABEL_THRESHOLD


def test_predict_multilabel_single_and_compound():
    res = train_multilabel_model()
    model, vec, mlb = res["model"], res["vectorizer"], res["mlb"]

    # Single category test
    pred_scarcity = predict_multilabel("Only 2 items left in stock!", model, vec, mlb)
    assert "Scarcity" in pred_scarcity["detected_categories"]
    assert pred_scarcity["is_dark"] is True

    # Neutral test
    pred_neutral = predict_multilabel("This shirt is made of 100% cotton fabric.", model, vec, mlb)
    assert "Not Dark Pattern" in pred_neutral["detected_categories"]
    assert pred_neutral["is_dark"] is False

    # Compound test
    compound = "Hurry! Only 2 items left in stock. This offer expires in 5 minutes."
    pred_compound = predict_multilabel(compound, model, vec, mlb)
    assert pred_compound["is_dark"] is True
    assert "Scarcity" in pred_compound["detected_categories"]
    assert "Urgency" in pred_compound["detected_categories"]


def test_explain_multilabel():
    res = train_multilabel_model()
    model, vec, mlb = res["model"], res["vectorizer"], res["mlb"]

    compound = "Hurry! Only 2 items left in stock. This offer expires in 5 minutes."
    explanations = explain_multilabel(compound, model, vec, mlb, top_n=3)
    assert isinstance(explanations, dict)
    assert "Scarcity" in explanations
    assert "Urgency" in explanations
    assert len(explanations["Scarcity"]) > 0
    scarcity_words = [t[0] for t in explanations["Scarcity"]]
    assert any(w in scarcity_words for w in ["left", "stock", "only"])


def test_multilabel_probabilities_structure():
    res = train_multilabel_model()
    model, vec, mlb = res["model"], res["vectorizer"], res["mlb"]
    pred = predict_multilabel("Limited time offer!", model, vec, mlb)
    probs = pred["probabilities"]
    assert len(probs) == len(res["classes"])
    for cls in res["classes"]:
        assert cls in probs
        assert 0.0 <= probs[cls] <= 1.0


def test_triple_compound_prediction_and_explanation():
    res = train_multilabel_model()
    model, vec, mlb = res["model"], res["vectorizer"], res["mlb"]
    text = "Hurry! Only 2 left in stock. 40 people are viewing this right now!"
    pred = predict_multilabel(text, model, vec, mlb)
    assert pred["is_dark"] is True
    overlap = set(pred["detected_categories"]).intersection({"Scarcity", "Social Proof", "Urgency"})
    assert len(overlap) >= 2

    expl = explain_multilabel(text, model, vec, mlb, top_n=2)
    for cat in overlap:
        assert len(expl[cat]) > 0


def test_evaluate_on_multilabel_dataset_metrics():
    eval_res = evaluate_on_multilabel_dataset()
    assert eval_res["benchmark_samples"] >= 20
    assert eval_res["threshold"] == DEFAULT_MULTILABEL_THRESHOLD
    assert 0.50 <= eval_res["subset_accuracy"] <= 1.0
    assert 0.0 <= eval_res["hamming_loss"] <= 0.20
    assert 0.70 <= eval_res["micro_f1"] <= 1.0

    per_label = eval_res["per_label_metrics"]
    assert "Scarcity" in per_label
    assert "Social Proof" in per_label
    assert "Urgency" in per_label
    assert per_label["Scarcity"]["f1"] > 0.80
    assert per_label["Social Proof"]["f1"] > 0.80
    assert per_label["Urgency"]["f1"] > 0.80


def test_examples_with_multiple_correct_labels():
    res = train_multilabel_model()
    model, vec, mlb = res["model"], res["vectorizer"], res["mlb"]

    # Compound 1: Scarcity + Urgency
    text1 = "Hurry! Only 2 items left in stock. This flash sale ends in 10 minutes!"
    pred1 = predict_multilabel(text1, model, vec, mlb)
    assert "Scarcity" in pred1["detected_categories"]
    assert "Urgency" in pred1["detected_categories"]

    # Compound 2: Scarcity + Social Proof
    text2 = "Only 2 items left in stock! 28 people are viewing this property right now."
    pred2 = predict_multilabel(text2, model, vec, mlb)
    assert "Scarcity" in pred2["detected_categories"]
    assert "Social Proof" in pred2["detected_categories"]

    # Compound 3: Urgency + Social Proof
    text3 = "Hurry, flash deal ends in 5 minutes! 85 people are viewing this right now."
    pred3 = predict_multilabel(text3, model, vec, mlb)
    assert "Urgency" in pred3["detected_categories"]
    assert "Social Proof" in pred3["detected_categories"]


def test_multilabel_consistent_default_threshold():
    assert DEFAULT_MULTILABEL_THRESHOLD == 0.35
    res = train_multilabel_model()
    pred = predict_multilabel("Limited stock remaining", res["model"], res["vectorizer"], res["mlb"])
    assert pred["threshold"] == DEFAULT_MULTILABEL_THRESHOLD


def test_evaluate_multilabel_cli_run(tmp_path):
    from evaluate_multilabel import run_multilabel_evaluation
    out_file = tmp_path / "test_multilabel_report.md"
    run_multilabel_evaluation(output_path=str(out_file))
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Multi-Label Detection" in content
    assert "Multi-Label Benchmark Evaluation" in content
