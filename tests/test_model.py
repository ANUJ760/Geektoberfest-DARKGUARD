from model import (
    build_vectorizer,
    clean_text,
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

