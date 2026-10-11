from model import (
    clean_text,
    explain_multilabel,
    load_dataset,
    predict_multilabel,
    predict_text,
    train_model,
    train_multilabel_model,
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


def test_train_multilabel_model():
    res = train_multilabel_model()
    assert "model" in res
    assert "mlb" in res
    assert len(res["classes"]) == 8
    assert "Scarcity" in res["classes"]
    assert "Urgency" in res["classes"]
    assert "Social Proof" in res["classes"]
    assert "report_dict" in res


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
    pred_compound = predict_multilabel(compound, model, vec, mlb, threshold=0.35)
    assert pred_compound["is_dark"] is True
    assert any(c in pred_compound["detected_categories"] for c in ["Scarcity", "Urgency"])


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

