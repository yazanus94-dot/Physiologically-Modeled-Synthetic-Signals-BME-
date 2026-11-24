"""Regression tests for confirmed scientific-computing defects."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aami import segment_beats  # noqa: E402
from features import extract_features_from_beats  # noqa: E402
from load_mitbih import MITBIHLoader  # noqa: E402
from run_detector_driven import _matched_features  # noqa: E402


def test_record_info_uses_signal_length_not_channel_count(tmp_path):
    (tmp_path / "100.dat").touch()
    (tmp_path / "100.hea").touch()
    with patch("load_mitbih.wfdb") as mock_wfdb:
        record = MagicMock()
        record.fs = 360
        record.n_sig = 2
        record.sig_len = 3600
        record.sig_name = ["MLII", "V1"]
        ann = MagicMock()
        ann.sample = np.array([360, 720])
        ann.symbol = np.array(["N", "V"])
        mock_wfdb.rdrecord.return_value = record
        mock_wfdb.rdann.return_value = ann
        info = MITBIHLoader(data_dir=tmp_path).get_record_info("100")
    assert info["n_samples"] == 3600
    assert info["duration_s"] == 10.0


def test_r_amp_relative_measures_r_position():
    fs = 360
    signal = np.zeros(10 * fs)
    r_peaks = np.array([4 * fs])
    signal[r_peaks[0]] = 1.0
    beats, _ = segment_beats(signal, r_peaks, fs)
    assert beats.shape[0] == 1
    X, names = extract_features_from_beats(beats, r_peaks, fs)
    assert X[0][names.index("r_amp_rel")] > 0.95


def test_real_record_loader_filters_non_beat_annotations(tmp_path):
    with patch("wfdb.rdrecord") as mock_record, patch("wfdb.rdann") as mock_ann:
        record = MagicMock()
        record.fs = 360
        record.p_signal = np.zeros((3600, 2))
        mock_record.return_value = record
        ann = MagicMock()
        ann.symbol = ["N", "[", "V", "x"]
        ann.sample = np.array([10, 20, 30, 40])
        mock_ann.return_value = ann
        from generate_signal import load_real_mitbih_record

        _, peaks, symbols, _ = load_real_mitbih_record(str(tmp_path), "100")
    assert list(peaks) == [10, 30]
    assert symbols == ["N", "V"]


def test_unmatched_detection_count_excludes_boundary_detections():
    signal = np.zeros(3600)
    detections = np.array([100, 200, 300, 400])
    # Only detection index 1 has a usable AAMI label. Classifiable indices are
    # 1 and 2, so exactly one of them is unmatched (not two as the old
    # range(1, n_det) diagnostic reported).
    features, labels, names, unmatched, matched = _matched_features(
        signal, detections, {1: "N"}, 360
    )
    assert matched == 1
    assert unmatched == 1
