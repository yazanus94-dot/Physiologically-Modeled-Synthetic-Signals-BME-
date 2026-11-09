"""Fast unit tests for the core pieces of the ECG pipeline."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_signal import generate_ecg        # noqa: E402
from detect_rpeaks import detect_rpeaks, score_detector  # noqa: E402
from features import extract_features           # noqa: E402


def test_generator_shapes_and_labels():
    sig, peaks, labels, fs = generate_ecg(duration_s=30, fs=250, seed=1)
    assert fs == 250
    assert len(sig) == 30 * 250
    assert len(peaks) == len(labels)
    assert set(labels) <= {"N", "V", "S"}
    assert set(labels) == {"N", "V", "S"}  # all classes present
    assert np.all(np.diff(peaks) > 0)


def test_detector_on_clean_signal_is_sharp():
    sig, peaks_true, labels, fs = generate_ecg(duration_s=20, fs=250, seed=2, noise=False)
    det = detect_rpeaks(sig, fs)
    score = score_detector(det, peaks_true, fs, tolerance_s=0.05)
    # classic 5-15 Hz band: sharp R detection, high precision; wide PVC
    # morphologies are partly attenuated, which is the known realistic
    # behaviour (and documented limitation) of this band
    assert score["sensitivity"] > 0.80
    assert score["precision"] > 0.90


def test_features_shapes():
    sig, peaks, labels, fs = generate_ecg(duration_s=20, fs=250, seed=3)
    X, names = extract_features(sig, peaks, fs)
    assert X.shape[1] == 6
    assert len(names) == 6
    assert X.shape[0] == len(peaks) - 2  # first and last beats have no neighbours
