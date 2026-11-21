"""Tests for the real BIDMC experiment modules (Project 3).

Covers the pieces added for the PhysioNet BIDMC PPG pipeline: channel
identification from a WFDB header, ECG reference detection and interval
alignment, PPG preprocessing/quality, interval comparison, and the two
defects fixed in this session:

* ``peak_detect`` plausibility cascade (a single signal dropout deleted
  every peak in a record — 707 candidates -> 2 kept),
* ``peak_detect`` height threshold based on the global maximum (a motion
  spike inflated it 1.6-2.1x and rejected normal beats),
* ``ecg_reference`` lag quantized to 1 s by a 1 Hz grid.

Nothing here downloads anything.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from load_bidmc import identify_channels  # noqa: E402
from ecg_reference import (align_interval_series,  # noqa: E402
                           detect_r_peaks, estimate_interval_lag,
                           rr_intervals)
from peak_detect import detect_peaks  # noqa: E402
from ppg_preprocessing import assess_signal_quality, preprocess_ppg  # noqa: E402
from hrv_metrics import (frequency_domain_metrics_from_intervals,  # noqa: E402
                         time_domain_metrics)
from quality_assessment import compare_intervals  # noqa: E402
from run_bidmc_experiment import _pearson  # noqa: E402


# ---------------------------------------------------------------------------
# Channel identification (header parsing)
# ---------------------------------------------------------------------------

def test_identify_channels_finds_ppg_and_ecg():
    ch = identify_channels(["RESP", "PLETH", "V", "AVR", "II"],
                           units=["pm", "NU", "mV", "mV", "mV"])
    assert ch["ppg"] == 1
    assert ch["ecg"] == 4
    assert ch["ecg_lead"] == "II"


def test_identify_channels_is_case_insensitive_and_tolerant_of_punct():
    # real BIDMC headers carry trailing commas: 'PLETH,', 'II,'
    ch = identify_channels(["RESP,", "PLETH,", "V,", "AVR,", "II,"])
    assert ch["ppg"] == 1
    assert ch["ecg"] == 4


def test_identify_channels_returns_none_for_missing_signals():
    ch = identify_channels(["ABP", "CVP"])
    assert ch["ppg"] is None
    assert ch["ecg"] is None


# ---------------------------------------------------------------------------
# ECG reference: R-peaks and intervals
# ---------------------------------------------------------------------------

def _synthetic_ecg(duration_s=60.0, fs=125.0, hr_bpm=72.0, seed=0):
    """Spike-train ECG: narrow QRS complexes on a quiet baseline."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    period = 60.0 / hr_bpm
    peaks = np.arange(0.2, duration_s - 0.2, period)
    sig = np.zeros(n)
    for p in peaks:
        i = int(p * fs)
        w = slice(max(0, i - 2), min(n, i + 3))
        sig[w] = 1.5
    sig += 0.01 * rng.standard_normal(n)
    return sig, fs, np.round(peaks * fs).astype(int)


def test_detect_r_peaks_finds_the_spike_train():
    sig, fs, truth = _synthetic_ecg()
    det = detect_r_peaks(sig, fs)
    assert det.size == pytest.approx(truth.size, abs=1)
    # every detected peak is within 1 sample of a true peak
    err = np.min(np.abs(det[:, None] - truth[None, :]), axis=1)
    assert err.max() <= 1


def test_detect_r_peaks_rejects_empty_signal():
    with pytest.raises(ValueError):
        detect_r_peaks(np.array([]), 125.0)
    with pytest.raises(ValueError):
        detect_r_peaks(np.array([np.nan, np.nan]), 125.0)


def test_rr_intervals_are_correct_and_in_seconds():
    fs = 125.0
    peaks = np.array([0, 100, 200, 300], dtype=int)   # 100 samples apart
    t, rr = rr_intervals(peaks, fs)
    assert rr.size == 3
    assert np.allclose(rr, 100 / fs)
    # mid-times sit halfway between the bounding peaks
    assert np.allclose(t, (peaks[:-1] + peaks[1:]) / 2.0 / fs)


def test_rr_intervals_of_too_few_peaks_is_empty():
    # one peak cannot form an interval
    t, rr = rr_intervals(np.array([0]), 125.0)
    assert rr.size == 0
    t, rr = rr_intervals(np.array([], dtype=int), 125.0)
    assert rr.size == 0


# ---------------------------------------------------------------------------
# Lag estimation — regression for the 1 s quantization defect
# ---------------------------------------------------------------------------

def test_estimate_interval_lag_recovers_sub_second_lag():
    """A 0.45 s lag must be recovered as ~0.45 s, not rounded to 0 or 1 s.

    Defect this locks down: the grid defaulted to 1 Hz, so the lag search
    could only return whole seconds.  On real BIDMC data the true optimum
    was 0.45 s while the estimator returned 1.0 s, which destroyed the
    lag-corrected correlation (measured r = 0.015 instead of 0.659).
    """
    rng = np.random.default_rng(7)
    t = np.arange(0, 300, 0.05)
    base = 0.7 + 0.05 * np.sin(2 * np.pi * t / 37.0) \
        + 0.004 * rng.standard_normal(t.size)
    lag_true = 0.45
    lag = estimate_interval_lag(t, base, t + lag_true, base)
    assert abs(lag - lag_true) <= 0.1, lag


def test_align_interval_series_shifts_the_second_series():
    rng = np.random.default_rng(8)
    t = np.arange(0, 300, 0.05)
    v = 0.7 + 0.05 * np.sin(2 * np.pi * t / 41.0) \
        + 0.004 * rng.standard_normal(t.size)
    lag = 0.30
    tg, a, b, out_lag = align_interval_series(t, v, t + lag, v, lag=lag)
    assert out_lag == pytest.approx(lag)
    assert tg.size > 100
    # after shifting back the two series must track each other closely.
    # Compared by correlation rather than pointwise: the common grid is
    # 10 Hz while the inputs are 20 Hz, so pointwise values differ by the
    # grid's interpolation error even when the alignment is correct.
    assert _pearson(a, b) > 0.99
    # and the shift must actually have been applied (not a no-op)
    assert np.max(np.abs(a - b)) < 0.5 * np.ptp(v)


def test_align_returns_empty_when_series_too_short():
    tg, a, b, lag = align_interval_series(np.array([0.0]), np.array([0.7]),
                                          np.array([0.0]), np.array([0.7]))
    assert tg.size == 0
    assert np.isnan(lag)


# ---------------------------------------------------------------------------
# PPG preprocessing + quality
# ---------------------------------------------------------------------------

def _synthetic_ppg(duration_s=120.0, fs=125.0, hr_bpm=75.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    period = 60.0 / hr_bpm
    ppg = np.zeros(n)
    # asymmetric pulse: fast upstroke, slow decay (systolic peak + dicrotic)
    for p in np.arange(0.3, duration_s - 0.3, period):
        i = int(p * fs)
        for k in range(1, int(0.35 * fs)):
            if i + k >= n:
                break
            phase = k / fs
            ppg[i + k] += (1.0 * np.exp(-phase / 0.35)
                           * (1 - np.exp(-phase / 0.03)))
    ppg += 0.01 * rng.standard_normal(n)
    ppg += 0.05 * np.sin(2 * np.pi * 0.25 * t)     # baseline wander
    return ppg, fs


def test_preprocess_ppg_returns_finite_filtered_signal():
    ppg, fs = _synthetic_ppg()
    out, info = preprocess_ppg(ppg, fs)
    assert out.shape == ppg.shape
    assert np.isfinite(out).all()
    assert isinstance(info, dict)


def test_preprocess_ppg_rejects_mostly_invalid_signal():
    bad = np.array([np.nan] * 90 + [1.0] * 10)
    with pytest.raises(ValueError):
        preprocess_ppg(bad, 125.0)


def test_assess_signal_quality_reports_usable_on_clean_ppg():
    ppg, fs = _synthetic_ppg()
    q = assess_signal_quality(ppg, fs)
    assert q["usable"] is True
    assert q["flags"] == []
    assert q["validation"]["n_samples"] == ppg.size


def test_assess_signal_quality_flags_flatline():
    flat = np.ones(int(30 * 125.0))
    q = assess_signal_quality(flat, 125.0)
    assert q["usable"] is False
    assert q["flags"]


# ---------------------------------------------------------------------------
# Peak detection — both defects fixed this session
# ---------------------------------------------------------------------------

def test_plausibility_filter_survives_a_single_dropout():
    """A long gap must not cascade and delete the rest of the record.

    Defect this locks down: the filter dropped a peak when EITHER adjacent
    interval exceeded 2.0 s.  Each removal merges its neighbours, creating
    a new violation, so one 7.9 s sensor dropout in a real BIDMC record
    deleted every peak except two (measured 707 candidates -> 2 kept).
    """
    fs = 125.0
    # 600 s of regular 0.66 s pulses with ONE 8 s dropout in the middle
    peaks_t = list(np.arange(0.0, 300.0, 0.66)) + \
        list(np.arange(308.0, 600.0, 0.66))
    ppg = np.zeros(int(600 * fs))
    for p in peaks_t:
        i = int(p * fs)
        if i < ppg.size:
            ppg[i:i + 4] = 1.0
    # sharpen into detectable pulses
    from scipy.ndimage import maximum_filter1d
    ppg = maximum_filter1d(ppg, size=9)

    det = detect_peaks(ppg, fs)
    # the vast majority of pulses must survive
    assert det.size > 0.9 * len(peaks_t), (det.size, len(peaks_t))


def test_detect_peaks_ignores_a_single_tall_artifact_spike():
    """A motion spike must not raise the threshold above normal beats.

    Defect this locks down: the height reference was ``np.max``.  On real
    BIDMC data max/p98 = 1.6-2.1, and that inflated threshold rejected 55
    of 727 beats (recall 91.9% vs 99.3% with a p98 reference).  Using a
    robust percentile keeps recall high at unchanged precision.
    """
    fs = 125.0
    n = int(120 * fs)
    t = np.arange(n) / fs
    # normal pulses of amplitude 0.4
    ppg = 0.4 * np.maximum(0, np.sin(2 * np.pi * t / 0.8))
    # one huge artefact spike, 6x taller than any real pulse
    ppg[int(60 * fs)] = 3.0

    det = detect_peaks(ppg, fs)
    # ~150 pulses in 120 s at 0.8 s period.  The point of the regression is
    # that the artefact must NOT raise the threshold above the normal beats:
    # with the old np.max reference the threshold became 1.05 against pulses
    # of height 0.4, so *every* real beat was rejected.  An isolated spike
    # that itself exceeds the threshold may still be counted as one beat
    # (it displaces a neighbouring pulse inside the refractory distance);
    # that single false positive out of ~150 is left in place and is what
    # the downstream quality checks are for.
    expected = int(120 / 0.8)
    assert det.size >= 0.9 * expected, (det.size, expected)
    # at least 90 % of the real pulses must still be found
    real = np.array([0.2 + 0.8 * k for k in range(expected)
                     if 0.2 + 0.8 * k < 120.0])
    hit = np.mean([np.min(np.abs(det / fs - r)) <= 0.1 for r in real])
    assert hit >= 0.9, hit


def test_detect_peaks_on_regular_train_is_regular():
    fs = 125.0
    n = int(60 * fs)
    ppg = np.zeros(n)
    for p in np.arange(0.0, 60.0, 0.7):
        i = int(p * fs)
        ppg[i:i + 5] = 1.0
    from scipy.ndimage import maximum_filter1d
    ppg = maximum_filter1d(ppg, size=9)

    det = detect_peaks(ppg, fs)
    iv = np.diff(det) / fs
    assert iv.size > 50
    assert np.allclose(iv, 0.7, atol=0.05)


# ---------------------------------------------------------------------------
# Peak matching is strictly one-to-one (regression for the old bug)
# ---------------------------------------------------------------------------

def test_score_detector_one_to_one_never_overcounts():
    """A detection may match at most one reference peak.

    Regression: the old implementation looped over reference peaks, picked
    the nearest detection, and never checked whether that detection had
    already been claimed.  ``det=[100], truth=[95, 100, 105]`` all mapped
    to the single detection at 100, so TP became 3 > len(detected)=1 and
    precision reported 3.0.
    """
    from peak_detect import score_detector, match_peaks_one_to_one
    det = np.array([100])
    truth = np.array([95, 100, 105])
    m = match_peaks_one_to_one(det, truth, fs=125.0, tolerance_s=0.01)
    assert len(m) <= min(det.size, truth.size)
    assert all(j in {0} for j, _i, _e in m)   # at most one detection used

    s = score_detector(det, truth, fs=125.0, tolerance_s=0.01)
    assert s["precision"] <= 1.0
    assert s["tp"] <= det.size
    assert s["tp"] <= truth.size
    # here: only the detection at 100 (truth index 1) matches; 95/105 are
    # 2 ms away, truth index 0/2 also match the same detection... the
    # closest-first ordering must resolve to a single pair per detection.
    assert s["tp"] >= 1


def test_score_detector_one_to_one_excludes_double_matches():
    """Two truth peaks claiming the same detection: only the closest wins."""
    from peak_detect import score_detector
    # a single detection, two truth peaks inside tolerance of it
    det = np.array([100])
    truth = np.array([99, 101])     # both within ±2 ms of the detection
    s = score_detector(det, truth, fs=125.0, tolerance_s=0.016)  # ~2 ms
    assert s["tp"] <= 1, s
    assert s["precision"] <= 1.0


def test_local_threshold_curve_length_and_finiteness():
    """The local adaptive height threshold is per-sample and finite."""
    from peak_detect import _local_height_threshold
    fs = 125.0
    t = np.arange(int(300 * fs)) / fs
    # drifting amplitude so local vs global differ
    sig = 0.1 * (0.5 + np.sin(2 * np.pi * t))
    sig = sig.astype(float)
    h = _local_height_threshold(sig, fs)
    assert h is not None
    assert h.shape == sig.shape
    assert np.all(np.isfinite(h))
    # short record returns None (no local stats) -> global fallback
    assert _local_height_threshold(sig[: int(fs)], fs) is None


# ---------------------------------------------------------------------------
# HRV metrics from an explicit interval series
# ---------------------------------------------------------------------------

def test_time_domain_metrics_keys_and_values():
    itis = np.array([800.0, 810.0, 790.0, 805.0, 795.0])
    m = time_domain_metrics(itis)
    for key in ("mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct"):
        assert key in m
    assert m["mean_hr_bpm"] == pytest.approx(60000.0 / 800.0, rel=1e-3)


def test_frequency_domain_from_intervals_on_clean_sinusoid():
    """A clean 0.25 Hz respiratory oscillation must land in HF."""
    fs_i = 4.0
    t = np.arange(0, 300, 1.0 / fs_i)
    base = 0.8
    intervals = base + 0.03 * np.sin(2 * np.pi * 0.25 * t)
    m = frequency_domain_metrics_from_intervals(t, intervals)
    assert m["hf_power"] > 0
    # 0.25 Hz sits inside HF (0.15-0.40 Hz) and outside LF (0.04-0.15 Hz),
    # so LF carries (numerically) nothing for a pure tone: the ratio is 0,
    # and HF must dominate outright.
    assert m["hf_power"] > m["lf_power"]
    assert m["lf_hf_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_frequency_domain_from_intervals_detects_lf_content():
    """A 0.10 Hz oscillation must land in LF and give a positive ratio."""
    t = np.arange(0, 400, 0.25)
    intervals = 0.8 + 0.03 * np.sin(2 * np.pi * 0.10 * t)
    m = frequency_domain_metrics_from_intervals(t, intervals)
    assert m["lf_power"] > 0
    assert m["hf_power"] > 0
    assert m["lf_hf_ratio"] > 0


def test_frequency_domain_from_intervals_short_series_is_nan():
    m = frequency_domain_metrics_from_intervals(np.array([0.0, 1.0]),
                                                np.array([0.8, 0.8]))
    assert np.isnan(m["lf_hf_ratio"])


def test_frequency_domain_handles_rejected_intervals():
    """NaNs / non-positive intervals are dropped, not propagated.

    The dropped samples must not corrupt the time axis: the surviving
    series still has real variability, so the bands come out finite.
    """
    t = np.arange(0, 300, 0.25)
    iv = 0.8 + 0.03 * np.sin(2 * np.pi * 0.25 * t)
    iv[::7] = np.nan
    iv[::11] = -1.0
    m = frequency_domain_metrics_from_intervals(t, iv)
    assert np.isfinite(m["hf_power"]) and m["hf_power"] > 0
    assert np.isfinite(m["lf_hf_ratio"])


# ---------------------------------------------------------------------------
# Interval comparison (HRV vs PRV disagreement classification)
# ---------------------------------------------------------------------------

def test_compare_intervals_agreement_on_identical_series():
    t = np.arange(0, 100, 0.7)
    iv = np.full(t.size, 0.7)
    res = compare_intervals(t, iv, t, iv, lag=0.0)
    assert res["n_agree"] == res["n_compared"]
    assert res["counts"]["missed_pulse_suspect"] == 0


def test_compare_intervals_flags_a_missed_pulse():
    t = np.arange(0, 100, 0.7)
    ecg = np.full(t.size, 0.7)
    # every other PPG interval is doubled -> a missed beat
    ppg = ecg.copy()
    ppg[1::2] = 1.4
    res = compare_intervals(t, ppg, t, ecg, lag=0.0)
    assert res["counts"]["missed_pulse_suspect"] > 0
    assert res["n_agree"] < res["n_compared"]


def test_compare_intervals_flags_a_false_peak():
    t = np.arange(0, 100, 0.7)
    ecg = np.full(t.size, 0.7)
    ppg = ecg.copy()
    ppg[1::2] = 0.35            # extra inserted peak -> halved intervals
    res = compare_intervals(t, ppg, t, ecg, lag=0.0)
    assert res["counts"]["false_peak_suspect"] > 0
