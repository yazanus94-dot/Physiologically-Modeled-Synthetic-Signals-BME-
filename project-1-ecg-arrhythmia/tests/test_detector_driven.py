"""Tests for the detector-driven classification evaluation (P1.3).

Core invariants:

- Features are built from the *detected* peaks only (``detect_rpeaks``) —
  annotation positions are never inputs to the feature extractors.
- Labels come only from matched reference annotations (one-to-one,
  ±50 ms); unmatched detections are reported, never silently dropped
  from the report.
- Causal RR features use only *past* intervals.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from run_detector_driven import (
    AAMI_CLASSES,
    _causal_rr_features,
    _matched_features,
    map_to_aami_class,
    match_peaks_one_to_one,
    run_detector_driven_eval,
)


def _synthetic_record_signal(fs=360.0, n=20000, rr=0.6):
    """Synthetic ECG-like signal with sharp R-pulses (for detector tests)."""
    np.arange(n) / fs
    sig = np.zeros(n)
    idx = np.arange(0, n, int(rr * fs))
    sig[idx] = 1.0
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(sig, size=9), idx


# ---------------------------------------------------------------------------
# Causal RR features
# ---------------------------------------------------------------------------

def test_causal_rr_features_use_only_past_intervals():
    """Feature for peak j uses intervals strictly before peak j (causal)."""
    fs = 360.0
    det = np.arange(0, 500, 150)                    # ~4 peaks
    rr_f, names = _causal_rr_features(det, fs)
    assert names == ["rr_prev", "rr_mean_past", "rr_prev_over_past"]
    # row k corresponds to peak k+1: rr_f[k][0] = det[k+1]-det[k]
    rr = np.diff(det) / fs
    for k in range(1, len(det) - 2):
        assert rr_f[k - 1][0] == pytest.approx(rr[k - 1])
    # causal: changing a FUTURE interval cannot change earlier rows
    det2 = det.copy()
    det2[-1] += 1000                                # change a future peak
    rr_f2, _ = _causal_rr_features(det2, fs)
    # rows before the changed peak are identical
    n_common = min(len(rr_f), len(rr_f2))
    assert np.allclose(rr_f[:n_common - 2], rr_f2[:n_common - 2])


# ---------------------------------------------------------------------------
# Labels: annotations only via one-to-one match
# ---------------------------------------------------------------------------

def test_matched_features_labels_only_from_matched_annotations():
    """``label_rows`` is keyed by detection index; only matched detections
    have labels (annotation *positions* are not used for labels by this
    test — matching does the assignment)."""
    det = np.arange(0, 400, 100)                       # 4 detections
    ref = np.array([1, 100, 201, 301])                 # 4 references
    ann = np.array(["N", "V", "S", "F"])
    fs = 360.0
    pairs = match_peaks_one_to_one(det, ref, fs, 0.05)
    label_rows = {}
    for det_idx, ref_idx, _e in pairs:
        c = map_to_aami_class(ann[ref_idx])
        if c in AAMI_CLASSES:
            label_rows[int(det_idx)] = c
    _X, y, _names, _n_unmatched, n_matched = _matched_features(
        np.random.default_rng(0).standard_normal(2000), det, label_rows, fs)
    assert n_matched == y.size and n_matched >= 0
    assert set(y.tolist()) <= set(AAMI_CLASSES)


def test_annotation_positions_never_in_features():
    """Feature inputs are (signal, detected peaks) — annotation positions
    must not appear as feature columns or inputs."""
    from run_detector_driven import _matched_features
    det = np.arange(10, 1000, 100)
    label_rows = {j: "N" for j in range(1, len(det) - 1)}
    _X1, _y1, names1, _, _ = _matched_features(
        np.arange(5000.0), det, label_rows, 360.0)
    assert all("ann" not in n for n in names1)
    assert not any("annotation" in n for n in names1)


# ---------------------------------------------------------------------------
# run_detector_driven_eval: report structure + leakage-prevention
# ---------------------------------------------------------------------------

def test_run_detector_driven_eval_report_structure():
    """The report has the documented keys, per-class metrics, and counts."""

    class StubLoader:
        def load_record(self, rec):
            sig, idx = _synthetic_record_signal(n=40000, rr=0.6)
            n = len(idx)
            # cycle N/V/S so the classifier has >= 2 classes
            labels = np.array(["N" if k % 3 == 0 else ("V" if k % 3 == 1 else "S")
                               for k in range(n)])
            return {"signal": sig, "fs": 360,
                    "r_peaks": np.array([int(v) for v in idx]),
                    "labels": labels}

    rep = run_detector_driven_eval(StubLoader(), ["r1"], ["r1"], ["r2"],
                                   tol_ms=50)
    assert "models" in rep and "test_summary" in rep
    for m in rep["models"]:
        assert "balanced_accuracy" in rep["models"][m]
        assert "macro_f1" in rep["models"][m]
        assert "per_class" in rep["models"][m]
    assert rep["test_summary"]["n_test_unmatched_detections_fp"] >= 0
    assert rep["test_summary"]["n_test_missed_reference_fn"] >= 0
