"""Regression tests for run_pipeline.py reporting and output wiring.

Two production defects are locked down here:

1. ``classification_report(..., target_names=...)`` without ``labels=``.
   sklearn infers ``labels`` by sorting the observed classes, so with
   AAMI_CLASSES = ['N','S','V','F'] the printed names were paired against
   ['F','N','S','V'] and every class name/support was silently shifted.
2. ``_save_mitbih_results`` writing into ``outputs/mitbih/<protocol>/``
   without creating it first -> FileNotFoundError on a fresh clone.
"""
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import run_pipeline as rp  # noqa: E402
from preprocessing import PreprocessingConfig  # noqa: E402

SRC_DIR = Path(rp.__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. target_names must always be paired with an explicit labels order
# ---------------------------------------------------------------------------

def _classification_report_calls():
    """Return (filename, call_source) for every classification_report(...) call."""
    calls = []
    for py in sorted(SRC_DIR.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for m in re.finditer(r"classification_report\s*\(", text):
            depth = 0
            i = m.end() - 1
            while i < len(text):
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            calls.append((py.name, text[m.start():i + 1]))
    return calls


def test_classification_report_target_names_paired_with_labels():
    """Every classification_report call must pass `labels` with `target_names`."""
    calls = _classification_report_calls()
    assert calls, "expected at least one classification_report call in src/"

    offenders = [f"{name}: {call}" for name, call in calls
                 if "target_names" in call and "labels" not in call]
    assert not offenders, (
        "classification_report called with target_names but without labels. "
        "sklearn pairs target_names positionally with its own sorted label "
        "order, so every printed class name is shifted:\n"
        + "\n\n".join(offenders)
    )


def test_aami_class_order_is_not_alphabetical():
    """The bug only bites because AAMI order != alphabetical order.

    If someone 'fixed' this by alphabetising AAMI_CLASSES the static test
    above would stop protecting anything - assert the orders really differ.
    """
    assert rp.AAMI_CLASSES == ["N", "S", "V", "F"]
    assert rp.AAMI_CLASSES != sorted(rp.AAMI_CLASSES)


# ---------------------------------------------------------------------------
# 2. results must be written into a directory that exists
# ---------------------------------------------------------------------------

class _StubModel:
    """Picklable stand-in exposing what get_feature_importance() needs."""

    def __init__(self, n_features):
        self.feature_importances_ = np.full(n_features, 1.0 / n_features)


@pytest.fixture
def mitbih_out(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "MITBIH_OUT", tmp_path)
    return tmp_path


def test_save_mitbih_results_creates_output_dir(mitbih_out):
    """Saving must create outputs/mitbih/<protocol>/ instead of crashing."""
    feat_names = ["rr_prev", "qrs_amp"]
    rf_eval = {"balanced_accuracy": 0.9, "macro_f1": 0.8,
               "weighted_f1": 0.85, "per_class": {}, "confusion_matrix": []}
    stub = _StubModel(len(feat_names))

    rp._save_mitbih_results(
        protocol="classic",
        split=SimpleNamespace(description="test split"),
        fs_target=360,
        preprocess_config=PreprocessingConfig(),
        rf_model=stub, rf_train_info={}, rf_eval=rf_eval, rf_per_record={},
        lr_model=stub, lr_train_info={}, lr_eval=rf_eval, lr_per_record={},
        train_counts={"N": 1}, val_counts={}, test_counts={"N": 1},
        train_records=["100"], val_records=["101"], test_records=["102"],
        feat_names=feat_names,
    )

    out = mitbih_out / "classic"
    assert out.is_dir(), "output directory was not created"
    assert (out / "metrics.json").is_file()
    assert (out / "rf_model.joblib").is_file()
    assert (out / "lr_model.joblib").is_file()

    with open(out / "metrics.json") as f:
        results = json.load(f)
    assert results["feature_names"] == feat_names
    assert results["protocol"] == "classic"
    # feature importances must be present for both models
    assert results["random_forest"]["feature_importances"]
    assert results["logistic_regression"]["feature_importances"]


def test_save_mitbih_results_roundtrips_model(mitbih_out):
    """The saved model must load back with its metadata."""
    from train_model import load_model

    feat_names = ["a", "b"]
    stub = _StubModel(2)
    rp._save_mitbih_results(
        protocol="leakage_audited",
        split=SimpleNamespace(description="d"),
        fs_target=360,
        preprocess_config=PreprocessingConfig(),
        rf_model=stub, rf_train_info={}, rf_eval={}, rf_per_record={},
        lr_model=stub, lr_train_info={}, lr_eval={}, lr_per_record={},
        train_counts={}, val_counts={}, test_counts={},
        train_records=[], val_records=[], test_records=[],
        feat_names=feat_names,
    )

    model, meta = load_model(mitbih_out / "leakage_audited" / "rf_model.joblib")
    assert meta["protocol"] == "leakage_audited"
    assert meta["model_type"] == "random_forest"
    assert np.allclose(model.feature_importances_, [0.5, 0.5])


def test_plot_mitbih_results_creates_dir(mitbih_out):
    """Plotting must also create its target directory."""
    feat_names = ["a", "b"]
    stub = _StubModel(2)
    empty_eval = {"confusion_matrix": np.zeros((4, 4), dtype=int),
                  "per_class": {c: {"f1": 0.0} for c in rp.AAMI_CLASSES}}

    rp._plot_mitbih_results(
        empty_eval, empty_eval, stub, stub,
        X_test=np.zeros((4, 2)), y_test=np.array(["N", "S", "V", "F"]),
        test_rec_ids=["100"] * 4,
        out_dir=mitbih_out / "fresh", feat_names=feat_names,
    )
    out = mitbih_out / "fresh"
    assert out.is_dir()
    assert (out / "confusion_matrices.png").is_file()
    assert (out / "per_class_f1.png").is_file()


# ---------------------------------------------------------------------------
# 3. CLI surface
# ---------------------------------------------------------------------------

def test_main_exposes_rpeaks_mode():
    """The R-peak detector evaluation must be reachable from the CLI.

    Checked statically: invoking main() here would actually run the whole
    (multi-minute) experiment as a side effect of collecting tests.
    """
    text = (SRC_DIR / "run_pipeline.py").read_text(encoding="utf-8")
    assert '"rpeaks"' in text, "--mode rpeaks missing from argparse choices"
    assert 'args.mode == "rpeaks"' in text, "rpeaks mode never dispatched"
    # and the experiment function it dispatches to must exist
    assert hasattr(rp, "run_rpeak_detection_experiment")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
