from model import (
    clean_text,
    evaluate_model,
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

