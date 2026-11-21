"""Tests for model training."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_model import (
    ModelConfig, create_model, train_model, evaluate_model,
    evaluate_per_record, get_feature_importance
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pytest


def test_model_config_defaults():
    """Test default model configuration."""
    config = ModelConfig()
    assert config.model_type == "random_forest"
    assert config.random_state == 42
    assert config.n_estimators == 200
    assert config.max_depth == 15
    assert config.class_weight == "balanced"


def test_model_config_serialization():
    """Test config serialization."""
    config = ModelConfig(model_type="logistic_regression", C=0.5)
    d = config.to_dict()
    assert d["model_type"] == "logistic_regression"
    assert d["C"] == 0.5

    config2 = ModelConfig.from_dict(d)
    assert config2.model_type == "logistic_regression"
    assert config2.C == 0.5


def test_create_model_rf():
    """Test Random Forest model creation."""
    config = ModelConfig(model_type="random_forest", n_estimators=50, max_depth=5)
    model = create_model(config)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 50
    assert model.max_depth == 5
    assert model.class_weight == "balanced"


def test_create_model_lr():
    """Test Logistic Regression model creation."""
    config = ModelConfig(model_type="logistic_regression", C=0.5, max_iter=500)
    model = create_model(config)
    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "clf" in model.named_steps
    assert isinstance(model.named_steps["clf"], LogisticRegression)
    assert model.named_steps["clf"].C == 0.5


def test_create_model_invalid():
    """Test invalid model type raises error."""
    config = ModelConfig(model_type="invalid_model")
    try:
        create_model(config)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_train_model_rf():
    """Test training Random Forest."""
    # Simple synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice(["N", "V", "S"], size=100, p=[0.7, 0.15, 0.15])

    config = ModelConfig(model_type="random_forest", n_estimators=20, max_depth=3)
    model, info = train_model(X, y, config)

    assert hasattr(model, "predict")
    assert info["n_train_samples"] == 100
    assert info["n_features"] == 5
    assert set(info["classes"]) == {"N", "V", "S"}


def test_train_model_lr():
    """Test training Logistic Regression."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice(["N", "V", "S"], size=100, p=[0.7, 0.15, 0.15])

    config = ModelConfig(model_type="logistic_regression", C=1.0, max_iter=500)
    model, info = train_model(X, y, config)

    assert hasattr(model, "predict")
    assert info["n_train_samples"] == 100


def test_evaluate_model():
    """Test model evaluation."""
    np.random.seed(42)
    X_train = np.random.randn(80, 5)
    y_train = np.random.choice(["N", "V", "S"], size=80, p=[0.7, 0.15, 0.15])
    X_test = np.random.randn(20, 5)
    y_test = np.random.choice(["N", "V", "S"], size=20, p=[0.7, 0.15, 0.15])

    config = ModelConfig(model_type="random_forest", n_estimators=20, max_depth=3)
    model, _ = train_model(X_train, y_train, config)

    eval_result = evaluate_model(model, X_test, y_test, class_names=["N", "V", "S"])

    assert "classification_report" in eval_result
    assert "confusion_matrix" in eval_result
    assert "balanced_accuracy" in eval_result
    assert "macro_f1" in eval_result
    assert "weighted_f1" in eval_result
    assert "per_class" in eval_result
    assert eval_result["balanced_accuracy"] >= 0
    assert eval_result["macro_f1"] >= 0
    assert len(eval_result["per_class"]) == 3


def test_evaluate_per_record():
    """Test per-record evaluation."""
    np.random.seed(42)
    X = np.random.randn(60, 5)
    y = np.random.choice(["N", "V", "S"], size=60, p=[0.7, 0.15, 0.15])
    record_ids = ["100"] * 20 + ["101"] * 20 + ["102"] * 20

    config = ModelConfig(model_type="random_forest", n_estimators=20, max_depth=3)
    model, _ = train_model(X[:40], y[:40], config)

    # Test on last 20 samples (all from record "102")
    eval_result = evaluate_per_record(model, X[40:], y[40:], record_ids[40:], class_names=["N", "V", "S"])

    assert "per_record" in eval_result
    assert len(eval_result["per_record"]) == 1  # Only record "102" in test set
    assert "102" in eval_result["per_record"]
    assert "balanced_accuracy" in eval_result["per_record"]["102"]


def test_get_feature_importance_rf():
    """Test feature importance for Random Forest."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice(["N", "V", "S"], size=100, p=[0.7, 0.15, 0.15])

    config = ModelConfig(model_type="random_forest", n_estimators=20, max_depth=3)
    model, _ = train_model(X, y, config)

    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    imp = get_feature_importance(model, feat_names)

    assert len(imp) == 5
    assert all(v >= 0 for v in imp.values())
    assert abs(sum(imp.values()) - 1.0) < 1e-6


def test_get_feature_importance_lr():
    """Test feature importance for Logistic Regression."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice(["N", "V", "S"], size=100, p=[0.7, 0.15, 0.15])

    config = ModelConfig(model_type="logistic_regression", C=1.0, max_iter=500)
    model, _ = train_model(X, y, config)

    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    imp = get_feature_importance(model, feat_names)

    assert len(imp) == 5
    assert all(v >= 0 for v in imp.values())


def test_model_reproducibility():
    """Test that same seed produces same model."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice(["N", "V", "S"], size=100, p=[0.7, 0.15, 0.15])

    config = ModelConfig(model_type="random_forest", n_estimators=20, max_depth=3, random_state=42)
    model1, _ = train_model(X, y, config)
    model2, _ = train_model(X, y, config)

    # Predictions should be identical
    preds1 = model1.predict(X)
    preds2 = model2.predict(X)
    assert np.array_equal(preds1, preds2)


class _StubModel:
    """Duck-typed model so evaluation tests are exact and deterministic.

    Predictions are looked up from column 0 of X, which holds each row's
    original index — so predicting on a row subset (as evaluate_per_record
    does) returns exactly the corresponding slice, like a real estimator.
    """

    def __init__(self, classes):
        self._classes = np.asarray(classes)

    def predict(self, X):
        idx = np.asarray(X)[:, 0].astype(int)
        return self._classes[idx]


def _row_index_X(n):
    """Feature matrix whose column 0 encodes the original row index."""
    return np.arange(n, dtype=float).reshape(-1, 1)


def test_report_class_names_align_with_supports():
    """target_names must be paired with the explicit label order.

    sklearn sorts the observed classes alphabetically when `labels` is not
    given, so ['N','S','V','F'] was silently mapped onto ['F','N','S','V'] and
    every printed class name was shifted by one position — the class with the
    largest support (N) was reported under the name 'S'.
    """
    class_names = ["N", "S", "V", "F"]          # AAMI order, NOT alphabetical
    y_true = (["N"] * 44) + (["S"] * 18) + (["V"] * 3) + ["F"]
    y_pred = list(y_true)
    y_pred[0] = "S"                              # one N misclassified as S
    y_pred[44] = "N"                             # one S misclassified as N

    model = _StubModel(y_pred)
    X = _row_index_X(len(y_true))

    res = evaluate_model(model, X, np.array(y_true), class_names)

    # Supports must follow class_names order, not alphabetical order
    assert res["classification_report"]["N"]["support"] == 44
    assert res["classification_report"]["S"]["support"] == 18
    assert res["classification_report"]["V"]["support"] == 3
    assert res["classification_report"]["F"]["support"] == 1

    # The report, the per-class dict and the confusion matrix must agree
    for cls in class_names:
        assert res["per_class"][cls]["support"] == \
            res["classification_report"][cls]["support"]
        assert res["per_class"][cls]["precision"] == pytest.approx(
            res["classification_report"][cls]["precision"])
        assert res["per_class"][cls]["recall"] == pytest.approx(
            res["classification_report"][cls]["recall"])

    cm = np.array(res["confusion_matrix"])
    assert [int(v) for v in np.diag(cm)] == [43, 17, 3, 1]
    assert cm.sum() == len(y_true)


def test_report_keys_follow_class_names_order():
    """Report entries are emitted in exactly the requested label order."""
    class_names = ["N", "S", "V", "F"]
    y = np.array(["N", "S", "V", "F", "N", "S"])
    model = _StubModel(["N", "S", "V", "F", "N", "S"])

    res = evaluate_model(model, _row_index_X(6), y, class_names)
    reported = [k for k in res["classification_report"] if k in class_names]
    assert reported == class_names


def test_per_record_evaluation_tolerates_missing_class():
    """A record lacking one of the four classes must not raise.

    Before the fix, classification_report() received 4 target_names for the
    3 classes present in that record and crashed with
    "Number of classes, 3, does not match size of target_names, 4".
    """
    class_names = ["N", "S", "V", "F"]

    # record A: F absent (3 classes)   |   record B: all 4 classes
    y_true = np.array(["N", "S", "V", "N", "S",
                       "N", "S", "V", "F", "N"])
    y_pred = np.array(["N", "S", "S", "N", "V",
                       "N", "S", "V", "F", "N"])
    record_ids = ["A"] * 5 + ["B"] * 5
    X = _row_index_X(10)

    model = _StubModel(y_pred)
    res = evaluate_per_record(model, X, y_true, record_ids, class_names)

    assert "A" in res["per_record"] and "B" in res["per_record"]
    # Absent class reports zero support rather than crashing
    assert res["per_record"]["A"]["per_class"]["F"]["support"] == 0
    assert res["per_record"]["B"]["per_class"]["F"]["support"] == 1
    assert res["per_record"]["A"]["class_counts"]["F"] == 0


def test_per_record_skips_single_class_records():
    """Records containing a single class are skipped, not scored."""
    class_names = ["N", "S", "V", "F"]
    y_true = np.array(["N", "N", "N", "N", "S", "V"])
    y_pred = np.array(["N", "N", "N", "N", "S", "V"])
    record_ids = ["single"] * 4 + ["multi"] * 2

    res = evaluate_per_record(_StubModel(y_pred), _row_index_X(6),
                              y_true, record_ids, class_names)
    assert "single" not in res["per_record"]
    assert "multi" in res["per_record"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])