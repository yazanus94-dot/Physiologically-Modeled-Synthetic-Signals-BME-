"""Fast unit tests for the core pieces of the ECG pipeline."""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_signal import generate_ecg        # noqa: E402
import pytest  # noqa: E402  (must follow sys.path setup)
from detect_rpeaks import detect_rpeaks, detect_rpeaks_mitbih, score_detector, score_detector_detailed  # noqa: E402
from features import extract_features, extract_rr_features, extract_qrs_morphology_features, extract_features_from_beats  # noqa: E402
from aami import (
    map_to_aami_class, filter_aami_beats,
    segment_beats_annotation_centered, count_beats_per_class,
    get_aami_classes, AAMI_MAPPING
)  # noqa: E402
from preprocessing import (
    preprocess_ecg, PreprocessingConfig,
    check_signal_quality, remove_dc_offset,
    design_bandpass_sos, apply_bandpass_filter
)  # noqa: E402
from load_mitbih import (
    MITBIHLoader, MITBIH_RECORDS, DS1_TRAIN_RECORDS, DS2_TEST_RECORDS,
    SAME_SUBJECT_PAIRS, PREFERRED_LEADS
)  # noqa: E402
from evaluation import (
    get_classic_split, get_leakage_audited_split,
    verify_no_leakage, get_subject_for_record,
    split_records_by_subject
)  # noqa: E402
from train_model import ModelConfig, create_model, train_model, evaluate_model, get_feature_importance  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# GENERATOR TESTS
# ============================================================

def test_generator_shapes_and_labels():
    sig, peaks, labels, fs = generate_ecg(duration_s=30, fs=250, seed=1)
    assert fs == 250
    assert len(sig) == 30 * 250
    assert len(peaks) == len(labels)
    assert set(labels) <= {"N", "V", "S"}
    assert set(labels) == {"N", "V", "S"}  # all classes present
    assert np.all(np.diff(peaks) > 0)


def test_generator_reproducibility():
    """Same seed should produce identical output."""
    sig1, peaks1, labels1, fs1 = generate_ecg(duration_s=30, fs=250, seed=42)
    sig2, peaks2, labels2, fs2 = generate_ecg(duration_s=30, fs=250, seed=42)
    assert np.array_equal(sig1, sig2)
    assert np.array_equal(peaks1, peaks2)
    assert labels1 == labels2


def test_generator_noise_control():
    """Noise parameter should work."""
    sig_clean, peaks_c, labels_c, fs = generate_ecg(duration_s=10, fs=250, seed=1, noise=False)
    sig_noisy, peaks_n, labels_n, fs = generate_ecg(duration_s=10, fs=250, seed=1, noise=True)
    # Same peaks and labels, different signal
    assert np.array_equal(peaks_c, peaks_n)
    assert labels_c == labels_n
    assert not np.array_equal(sig_clean, sig_noisy)


# ============================================================
# DETECTOR TESTS
# ============================================================

def test_detector_on_clean_signal_is_sharp():
    sig, peaks_true, labels, fs = generate_ecg(duration_s=20, fs=250, seed=2, noise=False)
    det = detect_rpeaks(sig, fs)
    score = score_detector(det, peaks_true, fs, tolerance_s=0.05)
    # classic 5-15 Hz band: sharp R detection, high precision; wide PVC
    # morphologies are partly attenuated, which is the known realistic
    # behaviour (and documented limitation) of this band
    assert score["sensitivity"] > 0.80
    assert score["precision"] > 0.90


def test_detector_detailed_scoring():
    """Test detailed scoring with timing error."""
    sig, peaks_true, labels, fs = generate_ecg(duration_s=20, fs=250, seed=3, noise=False)
    det = detect_rpeaks(sig, fs)
    score = score_detector_detailed(det, peaks_true, fs, tolerance_s=0.05)
    assert "f1" in score
    assert "mean_timing_error_ms" in score
    assert "std_timing_error_ms" in score
    assert "matched_pairs" in score


def test_detector_mitbih_mode():
    """Test MIT-BIH optimized detector."""
    sig, peaks_true, labels, fs = generate_ecg(duration_s=20, fs=250, seed=4, noise=False)
    det = detect_rpeaks_mitbih(sig, fs)
    score = score_detector(det, peaks_true, fs, tolerance_s=0.05)
    assert score["sensitivity"] > 0.80
    assert score["precision"] > 0.90


def test_detector_refractory_period():
    """Test that refractory period prevents double detection."""
    # Create a signal with a very prominent peak that might trigger multiple detections
    fs = 250
    t = np.arange(5000) / fs
    sig = np.sin(2 * np.pi * 1.2 * t)  # 72 BPM
    # Add a very large spike
    sig[1000] += 10

    det = detect_rpeaks(sig, fs, refractory_s=0.20)
    # Check no two peaks within refractory period
    if len(det) > 1:
        diffs = np.diff(det) / fs
        assert np.all(diffs >= 0.20)


# ============================================================
# FEATURE EXTRACTION TESTS
# ============================================================

def test_features_shapes():
    sig, peaks, labels, fs = generate_ecg(duration_s=20, fs=250, seed=3)
    X, names = extract_features(sig, peaks, fs)
    assert X.shape[1] == 15  # 5 RR + 10 morphology
    assert len(names) == 15
    assert X.shape[0] == len(peaks) - 2  # first and last beats have no neighbours


def test_rr_features():
    """Test RR feature extraction separately."""
    peaks = np.array([250, 500, 750, 1000, 1250])  # 1 Hz = 60 BPM
    fs = 250
    X, names = extract_rr_features(peaks, fs)
    assert X.shape == (3, 5)  # 5 peaks - 2 = 3 with neighbors
    assert len(names) == 5
    assert "rr_prev" in names
    assert "rr_next" in names
    assert "rr_ratio" in names
    assert "prematurity_ratio" in names


def test_morphology_features():
    """Test QRS morphology feature extraction."""
    fs = 250
    # Simple synthetic QRS-like signal
    t = np.arange(750) / fs
    sig = np.zeros(5000)
    # Add some QRS complexes
    for r in [500, 1000, 1500, 2000, 2500]:
        qrs = np.exp(-((np.arange(-60, 61)) ** 2) / (2 * 15 ** 2))
        sig[r-60:r+61] += qrs
    peaks = np.array([500, 1000, 1500, 2000, 2500])

    X, names = extract_qrs_morphology_features(sig, peaks, fs)
    assert X.shape == (3, 10)  # 5 peaks - 2 = 3 with neighbors
    assert len(names) == 10
    assert "qrs_amp" in names
    assert "qrs_width" in names
    assert "beat_skew" in names
    assert "beat_kurtosis" in names


def test_features_from_beats():
    """Test feature extraction from pre-segmented beats."""
    fs = 360
    n_beats = 10
    beat_len = int(0.7 * fs)  # 700ms window
    beats = np.random.randn(n_beats, beat_len)
    r_peaks = np.arange(n_beats) * 360 + 180

    X, names = extract_features_from_beats(beats, r_peaks, fs)
    assert X.shape == (n_beats, 9)
    assert len(names) == 9


# ============================================================
# AAMI MAPPING TESTS
# ============================================================

def test_aami_mapping_normal():
    """Test normal beat mappings."""
    for sym in ["N", "L", "R", "e", "j"]:
        assert map_to_aami_class(sym) == "N"


def test_aami_mapping_supraventricular():
    """Test supraventricular beat mappings."""
    for sym in ["A", "a", "J", "S"]:
        assert map_to_aami_class(sym) == "S"


def test_aami_mapping_ventricular():
    """Test ventricular beat mappings."""
    for sym in ["V", "E"]:
        assert map_to_aami_class(sym) == "V"


def test_aami_mapping_fusion():
    """Test fusion beat mappings."""
    assert map_to_aami_class("F") == "F"


def test_aami_excluded_symbols():
    """Test that non-beat symbols are excluded."""
    for sym in ["[", "]", "(", ")", "x", "!", "?", "+", "~", "|"]:
        assert map_to_aami_class(sym) is None


def test_aami_unknown_symbol():
    """Test unknown symbols return None (not silently converted to N)."""
    assert map_to_aami_class("Q") is None
    assert map_to_aami_class("/") is None
    assert map_to_aami_class("f") is None


def test_filter_aami_beats():
    """Test filtering beats to valid AAMI classes."""
    r_peaks = np.array([100, 200, 300, 400, 500])
    labels = ["N", "V", "A", "F", "x"]  # x is excluded
    filtered_peaks, filtered_labels, indices = filter_aami_beats(r_peaks, labels)
    assert len(filtered_peaks) == 4
    assert filtered_labels == ["N", "V", "S", "F"]
    assert np.array_equal(indices, [0, 1, 2, 3])


def test_segment_beats_annotation_centered():
    """Test beat segmentation around annotations."""
    fs = 360
    signal = np.sin(np.arange(10000) * 2 * np.pi * 1.0 / fs)
    r_peaks = np.array([360, 720, 1080, 1440, 1800])
    labels = ["N", "V", "N", "A", "F"]

    beats, beat_labels, indices = segment_beats_annotation_centered(
        signal, r_peaks, labels, fs, pre_window_s=0.25, post_window_s=0.45
    )
    # All 5 beats should be valid (well within signal boundaries)
    # But "x" is excluded... wait, no "x" in labels
    # Actually all 5 are valid AAMI classes: N, V, N, A, F
    assert len(beats) == 5
    assert beat_labels == ["N", "V", "N", "S", "F"]
    assert beats.shape[1] == int(round(0.25 * fs)) + int(round(0.45 * fs)) + 1


def test_count_beats_per_class():
    """Test beat counting."""
    labels = ["N", "N", "V", "A", "F", "L", "R", "x", "!", "Q"]
    counts = count_beats_per_class(labels)
    assert counts["N"] == 4  # N, L, R, (j/e not present)
    assert counts["S"] == 1  # A
    assert counts["V"] == 1  # V
    assert counts["F"] == 1  # F
    assert counts["excluded"] == 3  # x, !, Q


# ============================================================
# PREPROCESSING TESTS
# ============================================================

def test_preprocessing_config():
    """Test PreprocessingConfig serialization."""
    config = PreprocessingConfig(lowcut_hz=0.5, highcut_hz=40.0, filter_order=4)
    d = config.to_dict()
    assert d["lowcut_hz"] == 0.5
    assert d["highcut_hz"] == 40.0

    config2 = PreprocessingConfig.from_dict(d)
    assert config2.lowcut_hz == 0.5


def test_check_signal_quality():
    """Test signal quality checking."""
    sig = np.array([1.0, 2.0, np.nan, 4.0, np.inf])
    # With default nan_policy="error", should raise
    try:
        check_signal_quality(sig, PreprocessingConfig(nan_policy="error"))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # With nan_policy="interpolate", should return report
    report = check_signal_quality(sig, PreprocessingConfig(nan_policy="interpolate"))
    assert report["has_nan"]
    assert report["has_inf"]
    assert report["nan_count"] == 1
    assert report["inf_count"] == 1


def test_remove_dc_offset():
    """Test DC offset removal."""
    sig = np.array([10.0, 11.0, 12.0, 10.0, 11.0])  # DC = 11
    sig_removed = remove_dc_offset(sig, method="median")
    assert abs(np.median(sig_removed)) < 1e-10


def test_design_bandpass_sos():
    """Test SOS filter design."""
    sos = design_bandpass_sos(0.5, 40.0, 360, order=4)
    assert sos.shape[1] == 6  # SOS format
    # 4th order bandpass = 4 sections (2 per order for bandpass)
    assert sos.shape[0] == 4


def test_apply_bandpass_filter():
    """Test bandpass filtering."""
    fs = 360
    t = np.arange(3600) / fs
    sig = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 50.0 * t)  # 1 Hz + 50 Hz
    filtered = apply_bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4)

    # 50 Hz should be attenuated
    from scipy.fft import fft, fftfreq
    fft_orig = np.abs(fft(sig))
    fft_filt = np.abs(fft(filtered))
    freqs = fftfreq(len(sig), 1/fs)

    # Find 50 Hz bin
    idx_50 = np.argmin(np.abs(freqs - 50))
    idx_1 = np.argmin(np.abs(freqs - 1))

    # 50 Hz should be much lower in filtered (attenuated to ~12%)
    assert fft_filt[idx_50] < fft_orig[idx_50] * 0.15
    # 1 Hz should be preserved (96%)
    assert fft_filt[idx_1] > fft_orig[idx_1] * 0.5


def test_preprocess_ecg_pipeline():
    """Test full preprocessing pipeline."""
    fs = 360
    t = np.arange(7200) / fs
    sig = (
        np.sin(2 * np.pi * 1.2 * t) +  # Heart beat
        0.1 * np.sin(2 * np.pi * 0.2 * t) +  # Baseline wander
        0.05 * np.sin(2 * np.pi * 50.0 * t) +  # Powerline
        0.02 * np.random.randn(len(t))  # Noise
    )

    config = PreprocessingConfig(
        lowcut_hz=0.5,
        highcut_hz=40.0,
        filter_order=4,
        remove_dc=True,
        normalize=False
    )

    processed, report = preprocess_ecg(sig, fs, config)
    assert len(processed) == len(sig)
    assert "steps" in report
    assert "bandpass_0.5-40.0Hz_order4_sos_zero_phase" in report["steps"]
    assert "dc_removal_median" in report["steps"]


def test_preprocess_with_nan():
    """Test preprocessing handles NaN with interpolation."""
    sig = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    fs = 250
    config = PreprocessingConfig(nan_policy="interpolate")
    processed, report = preprocess_ecg(sig, fs, config)
    assert not np.any(np.isnan(processed))


# ============================================================
# LOAD_MITBIH TESTS (without actual data)
# ============================================================

def test_mitbih_constants():
    """Test MIT-BIH constants are defined correctly."""
    # The database contains 48 records in total. 102, 104, 107 and 217 are
    # valid recordings that the standard benchmark leaves out of DS1/DS2.
    assert len(MITBIH_RECORDS) == 48
    assert len(set(MITBIH_RECORDS)) == 48
    assert "100" in MITBIH_RECORDS
    assert "234" in MITBIH_RECORDS
    assert set(DS1_TRAIN_RECORDS).issubset(MITBIH_RECORDS)
    assert set(DS2_TEST_RECORDS).issubset(MITBIH_RECORDS)
    assert len(DS1_TRAIN_RECORDS) == 22
    assert len(DS2_TEST_RECORDS) == 22
    assert SAME_SUBJECT_PAIRS["201"] == "202"
    assert SAME_SUBJECT_PAIRS["202"] == "201"
    assert "MLII" in PREFERRED_LEADS


def test_mitbih_loader_init():
    """Test loader initialization."""
    loader = MITBIHLoader(data_dir="/tmp/test_mitbih")
    assert loader.data_dir.exists()


# ============================================================
# EVALUATION TESTS
# ============================================================

def test_classic_split():
    """Test classic DS1/DS2 split."""
    split = get_classic_split()
    assert split.protocol == "classic"
    assert len(split.train_records) == 18  # 22 - 4 val
    assert len(split.val_records) == 4
    assert len(split.test_records) == 22
    # Check no overlap
    assert set(split.train_records).isdisjoint(split.val_records)
    assert set(split.train_records).isdisjoint(split.test_records)
    assert set(split.val_records).isdisjoint(split.test_records)


def test_leakage_audited_split():
    """Test leakage-audited split excludes 202 from test."""
    split = get_leakage_audited_split()
    assert split.protocol == "leakage_audited"
    assert "201" in split.train_records
    assert "202" not in split.test_records
    assert len(split.test_records) == 21  # 22 - 1


def test_verify_no_leakage_classic():
    """Classic split SHOULD have leakage (201/202 same subject)."""
    split = get_classic_split()
    result = verify_no_leakage(split.train_records, split.test_records, verbose=False)
    # Classic split has 201 in train, 202 in test - same subject!
    assert result is False


def test_verify_no_leakage_audited():
    """Leakage-audited split should have NO leakage."""
    split = get_leakage_audited_split()
    result = verify_no_leakage(split.train_records, split.test_records, verbose=False)
    assert result is True


def test_get_subject_for_record():
    """Test subject mapping."""
    assert get_subject_for_record("100") == "subject_100"
    assert get_subject_for_record("201") == "subject_201_202"
    assert get_subject_for_record("202") == "subject_201_202"


def test_split_records_by_subject():
    """Test random subject-level split."""
    records = ["100", "101", "103", "105", "106", "201", "202"]
    split = split_records_by_subject(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    # 201 and 202 should be in same set
    assert ("201" in split.train_records) == ("202" in split.train_records)
    assert ("201" in split.val_records) == ("202" in split.val_records)
    assert ("201" in split.test_records) == ("202" in split.test_records)
    # Verify no leakage
    assert verify_no_leakage(split.train_records, split.test_records, verbose=False)


# ============================================================
# MODEL TRAINING TESTS
# ============================================================

def test_model_config():
    """Test ModelConfig."""
    config = ModelConfig(model_type="random_forest", n_estimators=100)
    d = config.to_dict()
    assert d["model_type"] == "random_forest"
    assert d["n_estimators"] == 100

    config2 = ModelConfig.from_dict(d)
    assert config2.n_estimators == 100


def test_create_model_rf():
    """Test Random Forest model creation."""
    config = ModelConfig(model_type="random_forest", n_estimators=50, max_depth=5)
    model = create_model(config)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 50
    assert model.max_depth == 5


def test_create_model_lr():
    """Test Logistic Regression model creation."""
    config = ModelConfig(model_type="logistic_regression", C=0.5)
    model = create_model(config)
    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "clf" in model.named_steps


def test_train_and_evaluate():
    """Test full train/evaluate cycle with synthetic data."""
    from generate_signal import generate_ecg
    from detect_rpeaks import detect_rpeaks
    from features import extract_features
    from sklearn.model_selection import train_test_split

    sig, r_true, labels, fs = generate_ecg(duration_s=30, fs=250, seed=1)
    r_det = detect_rpeaks(sig, fs)
    X, feat_names = extract_features(sig, r_det, fs)

    # Align labels
    tol = int(0.05 * fs)
    y = []
    keep = []
    for i, p in enumerate(r_det):
        if i == 0 or i == len(r_det) - 1:
            continue
        j = np.argmin(np.abs(r_true - p))
        if abs(p - r_true[j]) <= tol:
            y.append(labels[j])
            keep.append(i - 1)

    X = X[keep]
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    config = ModelConfig(model_type="random_forest", n_estimators=50, max_depth=5)
    model, info = train_model(X_train, y_train, config)

    eval_result = evaluate_model(model, X_test, y_test, class_names=["N", "V", "S"])
    assert "balanced_accuracy" in eval_result
    assert "macro_f1" in eval_result
    assert "per_class" in eval_result
    assert eval_result["balanced_accuracy"] >= 0
    assert eval_result["macro_f1"] >= 0


def test_get_feature_importance():
    """Test feature importance extraction."""
    from generate_signal import generate_ecg
    from detect_rpeaks import detect_rpeaks
    from features import extract_features
    from sklearn.model_selection import train_test_split

    sig, r_true, labels, fs = generate_ecg(duration_s=20, fs=250, seed=1)
    r_det = detect_rpeaks(sig, fs)
    X, feat_names = extract_features(sig, r_det, fs)

    tol = int(0.05 * fs)
    y = []
    keep = []
    for i, p in enumerate(r_det):
        if i == 0 or i == len(r_det) - 1:
            continue
        j = np.argmin(np.abs(r_true - p))
        if abs(p - r_true[j]) <= tol:
            y.append(labels[j])
            keep.append(i - 1)

    X = X[keep]
    y = np.array(y)

    config = ModelConfig(model_type="random_forest", n_estimators=20, max_depth=3)
    model, _ = train_model(X, y, config)

    imp = get_feature_importance(model, feat_names)
    assert len(imp) == len(feat_names)
    assert all(v >= 0 for v in imp.values())
    assert abs(sum(imp.values()) - 1.0) < 1e-6


def test_one_to_one_matching_single_detection_cannot_match_two_truths():
    """A detection inside tolerance of two references counts as ONE match.

    The previous implementation set matched_true for every reference whose
    nearest detection was inside the window without checking whether that
    detection was already used, producing tp=2 from a single detection and
    precision = 2/1 = 2.0 (>100%).
    """
    fs = 100                       # tolerance 50 ms -> 5 samples
    truth = [1000, 1010]           # 100 ms apart, both within 5 of 1005
    detected = [1005]

    score = score_detector_detailed(detected, truth, fs, tolerance_s=0.05)

    assert score["tp"] == 1        # not 2
    assert score["fn"] == 1
    assert score["fp"] == 0
    assert score["sensitivity"] == pytest.approx(0.5)
    assert score["precision"] == pytest.approx(1.0)
    assert score["precision"] <= 1.0
    assert len(score["matched_pairs"]) == 1


def test_one_to_one_matching_duplicate_detections_count_as_fp():
    """Two detections around one reference -> one TP and one FP."""
    fs = 100
    truth = [1000]
    detected = [995, 1005]         # both inside +/-5 samples

    for scorer in (score_detector, score_detector_detailed):
        s = scorer(detected, truth, fs, tolerance_s=0.05)
        assert s["tp"] == 1
        assert s["fp"] == 1
        assert s["fn"] == 0
        assert s["sensitivity"] == pytest.approx(1.0)   # the one truth is found
        assert s["precision"] == pytest.approx(0.5)      # 1 TP / 2 detections
        assert s["precision"] <= 1.0


def test_matching_is_closest_first():
    """Greedy closest-pair assignment maximises correct pairings."""
    fs = 100
    truth = [100, 104]
    detected = [100, 102]

    score = score_detector_detailed(detected, truth, fs, tolerance_s=0.05)
    assert score["tp"] == 2
    assert score["fp"] == 0
    assert score["fn"] == 0


def test_precision_never_exceeds_one_on_pathological_input():
    """Stress case: precision and sensitivity must both stay within [0, 1]."""
    rng = np.random.default_rng(0)
    fs = 250
    truth = np.sort(rng.integers(0, 100000, size=400))
    # Cluster detections so many references share the same tolerance window
    detected = np.repeat(truth[::5], 4) + rng.integers(-3, 4, size=4 * len(truth[::5]))

    score = score_detector_detailed(detected, truth, fs, tolerance_s=0.05)
    assert 0.0 <= score["precision"] <= 1.0
    assert 0.0 <= score["sensitivity"] <= 1.0
    assert score["tp"] + score["fn"] == len(truth)
    assert score["tp"] + score["fp"] == len(detected)
    assert len(score["matched_pairs"]) == score["tp"]


def test_matched_pairs_are_unique_on_both_sides():
    """No detected index and no reference index appears twice."""
    fs = 360
    truth = np.arange(0, 5000, 700)
    rng_offsets = np.array([0, 2, -3, 100, -200])   # some far outside tolerance
    detected = truth + np.resize(rng_offsets, len(truth))

    score = score_detector_detailed(detected, truth, fs, tolerance_s=0.05)
    det_idx = [p[0] for p in score["matched_pairs"]]
    true_idx = [p[1] for p in score["matched_pairs"]]
    assert len(det_idx) == len(set(det_idx))
    assert len(true_idx) == len(set(true_idx))


def test_empty_inputs_score_zero():
    """No detections / no references must not divide by zero."""
    fs = 250
    assert score_detector([], [], fs)["sensitivity"] == 0.0
    assert score_detector([], [], fs)["precision"] == 0.0
    assert score_detector([100, 200], [], fs)["precision"] == 0.0
    assert score_detector([], [100, 200], fs)["sensitivity"] == 0.0


if __name__ == "__main__":
    # Run tests manually if needed
    import pytest
    pytest.main([__file__, "-v"])