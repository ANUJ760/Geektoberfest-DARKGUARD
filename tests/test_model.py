from model import (
    build_vectorizer,
    clean_text,
    evaluate_model,
    explain_prediction,
    load_dataset,
    predict_text,
    train_model,
)


def test_clean_text():
    assert clean_text("  HELLO    WORLD  ") == "hello world"


def test_dataset_has_expected_columns():
    df=load_dataset()
    assert "text" in df.columns
    assert "label" in df.columns
    assert len(df)>0


def test_model_trains():
    results=train_model()
    assert 0<=results["accuracy"]<=1
    assert results["feature_count"]>0


def test_prediction_returns_valid_class():
    results=train_model()
    prediction,confidence,probabilities=predict_text(
        "Hurry! Only 2 seats left!",
        results["model"],
        results["vectorizer"]
    )
    assert prediction in (0,1)
    assert 0<=confidence<=1
    assert len(probabilities)==2


def test_build_vectorizer():
    base_vec = build_vectorizer(use_enhanced=False)
    enh_vec = build_vectorizer(use_enhanced=True, char_min_df=1)
    sample = ["Only 2 left!", "Size: Medium"]
    X_base = base_vec.fit_transform(sample)
    X_enh = enh_vec.fit_transform(sample)
    assert X_base.shape[0] == 2
    assert X_enh.shape[0] == 2
    assert X_enh.shape[1] > X_base.shape[1]


def test_short_and_subtle_text_predictions():
    results = train_model(use_enhanced=True)
    model = results["model"]
    vectorizer = results["vectorizer"]

    # Short dark patterns should be predicted as 1
    for snippet in ["Only 2 left!", "Hurry!", "Limited time offer"]:
        pred, conf, _ = predict_text(snippet, model, vectorizer)
        assert pred == 1, f"Expected 1 for '{snippet}'"
        assert conf > 0.5

    # Short normal patterns should be predicted as 0
    for snippet in ["Size: Medium", "Color: Blue", "Standard shipping available"]:
        pred, conf, _ = predict_text(snippet, model, vectorizer)
        assert pred == 0, f"Expected 0 for '{snippet}'"
        assert conf > 0.5


def test_explain_prediction_strips_prefix():
    results = train_model(use_enhanced=True)
    pos_terms, neg_terms = explain_prediction(
        "Only 2 left!", results["model"], results["vectorizer"]
    )
    for term, score in pos_terms + neg_terms:
        assert not term.startswith("word__")
        assert not term.startswith("char__")


def test_short_subtle_eval_set_schema():
    import json
    from pathlib import Path
    eval_file = Path(__file__).resolve().parent / "short_subtle_eval_set.json"
    assert eval_file.exists()
    items = json.loads(eval_file.read_text(encoding="utf-8"))
    assert len(items) == 30
    for it in items:
        assert "text" in it and len(it["text"].strip()) > 0
        assert it["label"] in (0, 1)
        assert "category" in it
        assert "description" in it
    dark_count = sum(1 for it in items if it["label"] == 1)
    non_dark_count = sum(1 for it in items if it["label"] == 0)
    assert dark_count == 15
    assert non_dark_count == 15


def test_legacy_train_model_compatibility():
    res = train_model(use_enhanced=False)
    assert 0.90 <= res["accuracy"] <= 1.0
    pred, conf, _ = predict_text("Only 2 seats left!", res["model"], res["vectorizer"])
    assert pred in (0, 1)


def test_edge_case_inputs_prediction():
    results = train_model(use_enhanced=True)
    model, vec = results["model"], results["vectorizer"]

    # Single-word urgency
    pred_h, conf_h, _ = predict_text("Hurry!", model, vec)
    assert pred_h == 1

    # Single-word neutral
    pred_s, conf_s, _ = predict_text("Shoes", model, vec)
    assert pred_s == 0

    # Minimal countdown
    pred_c, conf_c, _ = predict_text("2 left", model, vec)
    assert pred_c == 1


def test_evaluate_short_text_cli_run(tmp_path):
    from evaluate_short_text import evaluate_models
    out_file = tmp_path / "test_short_report.md"
    evaluate_models(output_path=str(out_file))
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Short and Subtle Text Detection" in content
    assert "Benchmark Results" in content


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
