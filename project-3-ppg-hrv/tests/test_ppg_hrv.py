"""Fast unit tests for the PPG generator, detector, and HRV metrics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_ppg import generate_ppg              # noqa: E402
from peak_detect import detect_peaks, score_detector  # noqa: E402
from hrv_metrics import (time_domain_metrics,      # noqa: E402
                         _frequency_domain_metrics_from_peaks)


def test_generator_shapes_and_count():
    ppg, t, peaks, fs = generate_ppg(duration_s=20, fs=100, seed=7)
    assert fs == 100
    assert len(ppg) == 20 * 100
    # ~68 bpm over 20 s -> about 20-26 peaks
    assert 18 <= len(peaks) <= 27


def test_detector_on_clean_ppg():
    ppg, t, peaks_true, fs = generate_ppg(duration_s=30, fs=100, seed=11,
                                          sensor_noise_std=0.004, motion_bursts=0)
    det = detect_peaks(ppg, fs)
    score = score_detector(det, peaks_true, fs, tolerance_s=0.15)
    assert score["sensitivity"] > 0.9
    assert score["precision"] > 0.9


def test_time_metrics_regular_series():
    # a perfectly regular series at 60 bpm -> SDNN=0, RMSSD=0, pNN50=0
    peaks = np.arange(0, 100 * 60, 100)  # 1 s intervals at 100 Hz
    m = time_domain_metrics(np.diff(peaks) / 100 * 1000)
    assert m["sdnn_ms"] == 0.0
    assert m["rmssd_ms"] == 0.0
    assert m["pnn50_pct"] == 0.0
    assert abs(m["mean_hr_bpm"] - 60.0) < 0.01


def test_freq_metrics_keys():
    ppg, t, peaks, fs = generate_ppg(duration_s=60, fs=100, seed=13,
                                     motion_bursts=0)
    m = _frequency_domain_metrics_from_peaks(peaks, fs)
    assert {"vlf_power", "lf_power", "hf_power", "lf_hf_ratio"} == set(m)
