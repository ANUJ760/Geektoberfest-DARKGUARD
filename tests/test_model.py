from model import (
    clean_text,
    compare_classifiers,
    create_classifier,
    explain_prediction,
    load_dataset,
    predict_text,
    train_model,
)
import pytest


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


def test_create_classifier():
    lr = create_classifier("logistic_regression")
    svm = create_classifier("linear_svm")
    nb = create_classifier("naive_bayes")
    cnb = create_classifier("complement_nb")
    assert lr is not None
    assert svm is not None
    assert nb is not None
    assert cnb is not None

    with pytest.raises(ValueError):
        create_classifier("non_existent_classifier")


def test_train_model_with_different_classifiers():
    for cname in ["linear_svm", "naive_bayes"]:
        res = train_model(classifier=cname)
        assert 0.85 <= res["accuracy"] <= 1.0
        pred, conf, probs = predict_text("Only 2 left!", res["model"], res["vectorizer"])
        assert pred in (0, 1)
        assert 0.0 <= conf <= 1.0


def test_compare_classifiers():
    comp = compare_classifiers()
    df = comp["comparison_table"]
    assert len(df) == 4
    assert "Accuracy" in df.columns or "accuracy" in df.columns
    assert "recall" in df.columns
    assert "false_negatives" in df.columns
    assert len(comp["confusion_matrices"]) == 4


def test_explain_prediction_with_svm_and_nb():
    res_svm = train_model(classifier="linear_svm")
    pos_svm, neg_svm = explain_prediction("Hurry! Only 2 left!", res_svm["model"], res_svm["vectorizer"])
    assert isinstance(pos_svm, list)
    assert isinstance(neg_svm, list)

    res_nb = train_model(classifier="naive_bayes")
    pos_nb, neg_nb = explain_prediction("Hurry! Only 2 left!", res_nb["model"], res_nb["vectorizer"])
    assert isinstance(pos_nb, list)
    assert isinstance(neg_nb, list)


def test_complement_nb_training_and_prediction():
    res_cnb = train_model(classifier="complement_nb")
    assert 0.85 <= res_cnb["accuracy"] <= 1.0
    pred, conf, probs = predict_text("Only 2 seats left!", res_cnb["model"], res_cnb["vectorizer"])
    assert pred in (0, 1)
    assert 0.0 <= conf <= 1.0


def test_classifier_tradeoff_consistency():
    comp = compare_classifiers()
    df = comp["comparison_table"]
    # Check that comparison table has valid rows
    lr_row = df[df["classifier_key"] == "logistic_regression"].iloc[0]
    nb_row = df[df["classifier_key"] == "naive_bayes"].iloc[0]
    svm_row = df[df["classifier_key"] == "linear_svm"].iloc[0]

    # Verify key trade-offs:
    # NB recall should be higher than LR recall
    assert nb_row["recall"] >= lr_row["recall"]
    # NB false negatives should be strictly lower than LR false negatives
    assert nb_row["false_negatives"] < lr_row["false_negatives"]
    # Calibrated SVM should have fewer false negatives than LR
    assert svm_row["false_negatives"] < lr_row["false_negatives"]


def test_compare_classifiers_cli_run(tmp_path):
    from compare_classifiers import run_classifier_comparison
    out_file = tmp_path / "test_comparison_report.md"
    run_classifier_comparison(output_path=str(out_file))
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Classifier Comparison" in content
    assert "Linear SVM" in content
    assert "Confusion Matrices" in content


