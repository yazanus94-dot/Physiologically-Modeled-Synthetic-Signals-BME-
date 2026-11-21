"""Tests for feature extraction."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import (
    extract_features, extract_rr_features, extract_qrs_morphology_features,
    extract_features_from_beats, extract_rr_features_for_annotations,
    QRS_WINDOW_S
)
from aami import map_to_aami_class
import pytest


def test_extract_rr_features_basic():
    """Test basic RR feature extraction."""
    fs = 250
    # 5 peaks at 1 Hz (60 BPM)
    peaks = np.array([250, 500, 750, 1000, 1250])
    X, names = extract_rr_features(peaks, fs)

    assert X.shape == (3, 5)  # 5 peaks - 2 = 3 with both neighbors
    assert len(names) == 5
    assert names == ["rr_prev", "rr_next", "rr_mean_local", "rr_ratio", "prematurity_ratio"]

    # All RR intervals should be 1.0 second
    assert np.allclose(X[:, 0], 1.0)  # rr_prev
    assert np.allclose(X[:, 1], 1.0)  # rr_next
    assert np.allclose(X[:, 2], 1.0)  # rr_mean_local
    assert np.allclose(X[:, 3], 1.0)  # rr_ratio
    assert np.allclose(X[:, 4], 1.0)  # prematurity_ratio


def test_extract_rr_features_prematurity():
    """Test RR features capture prematurity."""
    fs = 250
    # Normal beat at 1s, then premature at 0.7s, then compensatory pause at 1.3s
    peaks = np.array([250, 500, 675, 1000, 1325])  # intervals: 1.0, 0.7, 1.3, 1.3
    X, names = extract_rr_features(peaks, fs)

    # For the premature beat (index 1 in X, which corresponds to peak at 675):
    # rr_prev = 0.7, rr_next = 1.3
    # rr_ratio = 1.3 / 0.7 ≈ 1.86
    assert X[1, 3] > 1.5  # rr_ratio should be high (compensatory pause after premature)


def test_extract_qrs_morphology_features():
    """Test QRS morphology features."""
    fs = 250
    # Create signal with clear QRS complexes
    sig = np.zeros(5000)
    peaks = np.array([500, 1000, 1500, 2000, 2500])

    # Add Gaussian-shaped QRS complexes
    for r in peaks:
        t = np.arange(-60, 61)
        qrs = np.exp(-(t ** 2) / (2 * 15 ** 2))  # Gaussian
        sig[r-60:r+61] += qrs

    X, names = extract_qrs_morphology_features(sig, peaks, fs)

    assert X.shape == (3, 10)  # 5 peaks - 2 = 3 with neighbors
    assert len(names) == 10

    # All features should be finite
    assert np.all(np.isfinite(X))

    # Amplitude should be positive
    assert np.all(X[:, 0] > 0)  # qrs_amp

    # Width should be positive and reasonable (around 120ms = 0.12s)
    assert np.all(X[:, 1] > 0)
    assert np.all(X[:, 1] < 0.3)  # Less than 300ms


def test_extract_features_combined():
    """Test combined feature extraction."""
    fs = 250
    sig = np.zeros(5000)
    peaks = np.array([500, 1000, 1500, 2000, 2500])

    for r in peaks:
        t = np.arange(-60, 61)
        qrs = np.exp(-(t ** 2) / (2 * 15 ** 2))
        sig[r-60:r+61] += qrs

    X, names = extract_features(sig, peaks, fs, include_rr=True, include_morphology=True)

    assert X.shape == (3, 15)  # 5 RR + 10 morphology
    assert len(names) == 15


def test_extract_features_rr_only():
    """Test RR-only feature extraction."""
    fs = 250
    sig = np.zeros(5000)
    peaks = np.array([500, 1000, 1500, 2000, 2500])

    X, names = extract_features(sig, peaks, fs, include_rr=True, include_morphology=False)

    assert X.shape == (3, 5)
    assert len(names) == 5


def test_extract_features_morphology_only():
    """Test morphology-only feature extraction."""
    fs = 250
    sig = np.zeros(5000)
    peaks = np.array([500, 1000, 1500, 2000, 2500])

    for r in peaks:
        t = np.arange(-60, 61)
        qrs = np.exp(-(t ** 2) / (2 * 15 ** 2))
        sig[r-60:r+61] += qrs

    X, names = extract_features(sig, peaks, fs, include_rr=False, include_morphology=True)

    assert X.shape == (3, 10)
    assert len(names) == 10


def test_extract_features_from_beats():
    """Test feature extraction from pre-segmented beats."""
    fs = 360
    n_beats = 10
    beat_len = int(0.7 * fs)  # 700ms window
    beats = np.random.randn(n_beats, beat_len)
    r_peaks = np.arange(n_beats) * 360 + 180

    X, names = extract_features_from_beats(beats, r_peaks, fs)

    assert X.shape == (n_beats, 9)
    assert len(names) == 9

    # All features should be finite
    assert np.all(np.isfinite(X))


def test_extract_features_insufficient_peaks():
    """Test handling of insufficient peaks."""
    fs = 250
    sig = np.random.randn(1000)
    peaks = np.array([100])  # Only 1 peak
    labels = ["N"]

    X, names = extract_features(sig, peaks, fs)
    assert X.shape == (0, 15)
    assert len(names) == 15


def test_qrs_window_size_constant():
    """Test QRS_WINDOW_S constant is reasonable."""
    assert QRS_WINDOW_S == 0.12  # 120 ms


# ---------------------------------------------------------------------------
# RR features for annotation-centred (AAMI) beats
# ---------------------------------------------------------------------------

def _ann_fixture(fs=360, interval_s=1.0, n_beats=6, symbols=None):
    """Build an annotation series with one beat every `interval_s` seconds."""
    samples = (np.arange(n_beats) * interval_s * fs).astype(np.int64)
    if symbols is None:
        symbols = ["N"] * n_beats
    return samples, symbols


def test_rr_for_annotations_basic_values():
    """RR features are correct for a regular 60 bpm annotation series."""
    fs = 360
    samples, symbols = _ann_fixture(fs=fs, n_beats=6)
    targets = np.array([1, 2, 3])

    X, names, keep = extract_rr_features_for_annotations(samples, symbols, targets, fs)

    assert names == ["rr_prev", "rr_next", "rr_mean_local",
                     "rr_ratio", "prematurity_ratio"]
    assert X.shape == (3, 5)
    assert keep.all()
    assert np.allclose(X[:, 0], 1.0)   # rr_prev
    assert np.allclose(X[:, 1], 1.0)   # rr_next
    assert np.allclose(X[:, 2], 1.0)   # rr_mean_local
    assert np.allclose(X[:, 3], 1.0)   # rr_ratio
    assert np.allclose(X[:, 4], 1.0)   # prematurity_ratio


def test_rr_for_annotations_detects_prematurity():
    """A premature beat produces rr_ratio > 1 (short RR then compensatory pause)."""
    fs = 360
    # intervals: 1.0, 0.7, 1.3 s  -> the beat at index 2 is premature
    samples = np.array([0, 360, 612, 1080], dtype=np.int64)
    symbols = ["N", "N", "V", "N"]

    X, _, keep = extract_rr_features_for_annotations(samples, symbols,
                                                      np.array([1, 2]), fs)
    assert keep.all()
    # beat 2: rr_prev = 0.7, rr_next = 1.3 -> rr_ratio ~ 1.86
    assert X[1, 0] == pytest.approx(0.7, abs=1e-6)
    assert X[1, 1] == pytest.approx(1.3, abs=1e-6)
    assert X[1, 3] == pytest.approx(1.3 / 0.7, rel=1e-6)


def test_rr_for_annotations_ignores_non_beat_symbols():
    """Rhythm markers and artefact flags must not split an RR interval."""
    fs = 360
    samples, symbols = _ann_fixture(fs=fs, n_beats=4)
    # Insert non-beat annotations between beats 0/1, 1/2 and 2/3
    samples = np.concatenate([samples, [180, 540, 900]])
    symbols = symbols + ["[", "]", "~"]
    order = np.argsort(samples)
    samples = samples[order]
    symbols = [symbols[i] for i in order]
    # Indices are into the *merged* series, exactly as the pipeline provides:
    # the AAMI beats now sit at positions 0, 2, 4, 6.
    beat_indices = np.array([2, 4])          # beats at 360 and 720 samples

    X, _, keep = extract_rr_features_for_annotations(samples, symbols,
                                                      beat_indices, fs)
    assert keep.all()
    assert np.allclose(X[:, 0], 1.0)   # intervals unaffected by non-beat markers
    assert np.allclose(X[:, 1], 1.0)


def test_rr_for_annotations_counts_paced_beats_as_beats():
    """A paced ('/') beat is not an AAMI class but still delimits RR."""
    fs = 360
    samples, symbols = _ann_fixture(fs=fs, n_beats=5)
    symbols[2] = "/"                      # paced beat, excluded from N/S/V/F
    aami_labels = [map_to_aami_class(s) for s in symbols]
    assert aami_labels[2] is None          # not classified ...

    X, _, keep = extract_rr_features_for_annotations(samples, symbols,
                                                      np.array([1, 2, 3]), fs)
    assert keep.all()                      # ... but RR context still exists
    assert np.allclose(X[:, 0], 1.0)
    assert np.allclose(X[:, 1], 1.0)


def test_rr_for_annotations_unknown_symbol_is_not_a_beat():
    """An unrecognised symbol must be skipped, not treated as a beat."""
    fs = 360
    samples = np.array([0, 360, 1080], dtype=np.int64)
    symbols = ["N", "Z", "N"]              # 'Z' is unknown -> not a beat

    X, _, keep = extract_rr_features_for_annotations(
        samples, symbols, np.arange(3), fs)

    # 'Z' is not a beat, so it has no RR context of its own. If it were
    # (wrongly) accepted as a beat, index 1 would be kept -> keep[1] would be
    # True. The two real beats sit at the ends of the series, so neither has
    # both a preceding and a following RR interval.
    assert list(keep) == [False, False, False]
    assert X.shape[0] == 0


def test_rr_for_annotations_boundary_beats_dropped():
    """First and last beats have no RR pair on both sides -> keep=False."""
    fs = 360
    samples, symbols = _ann_fixture(fs=fs, n_beats=5)

    X, _, keep = extract_rr_features_for_annotations(
        samples, symbols, np.arange(5), fs)

    assert list(keep) == [False, True, True, True, False]
    assert X.shape[0] == 3
    assert len(X) == int(keep.sum())


def test_rr_for_annotations_no_label_leakage():
    """Identical timing must yield identical features regardless of class label."""
    fs = 360
    samples, symbols_n = _ann_fixture(fs=fs, n_beats=6)
    _, symbols_v = _ann_fixture(fs=fs, n_beats=6)
    symbols_v = ["V"] * 6

    X_n, _, keep_n = extract_rr_features_for_annotations(samples, symbols_n,
                                                          np.array([1, 2, 3]), fs)
    X_v, _, keep_v = extract_rr_features_for_annotations(samples, symbols_v,
                                                          np.array([1, 2, 3]), fs)

    assert np.array_equal(keep_n, keep_v)
    assert np.allclose(X_n, X_v)          # labels never enter the features


def test_rr_for_annotations_empty_targets():
    """Empty target set returns an empty, well-formed result."""
    samples, symbols = _ann_fixture()
    X, names, keep = extract_rr_features_for_annotations(
        samples, symbols, np.array([], dtype=int), 360)
    assert X.shape == (0, 5)
    assert keep.size == 0
    assert len(names) == 5


def test_rr_for_annotations_respects_sampling_frequency():
    """Features are computed in seconds using the provided fs, not a constant."""
    targets = np.array([1, 2])
    out = {}
    for fs in (100, 250, 360, 500):
        samples, symbols = _ann_fixture(fs=fs, n_beats=5)  # 1 s spacing
        X, _, keep = extract_rr_features_for_annotations(samples, symbols, targets, fs)
        assert keep.all()
        out[fs] = X[0, 0]
    # A 1 s interval must be 1.0 s at every sampling frequency
    assert all(abs(v - 1.0) < 1e-9 for v in out.values())


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])