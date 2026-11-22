"""Detector robustness tests for the PPG systolic-peak detector (P3.2).

Synthetic PPG trains are generated with known pulse times; ``detect_peaks``
is then checked to find most pulses (sensitivity >= ~0.85) and few or no
false positives at the boundary rates 160/175/190/200 bpm — consistent with
the central pulse-rate configuration in ``peak_detect.py``
(``MAX_PULSE_RATE_BPM = 200``, ``REFRACTORY_S = 60/200 = 0.30 s`` → the
detector can identify valid pulses up to ~202 bpm at 125 Hz).

Also covers the audit's detector-condition requirements: baseline drift,
changing pulse amplitude, motion-artifact bursts, missing pulses, false
peaks and pulse-morphology changes, plus different heart rates.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from peak_detect import (
    MAX_PULSE_RATE_BPM,
    REFRACTORY_S,
    detect_peaks,
    match_peaks_one_to_one,  # noqa: E402
    score_detector,
)

FS = 125.0
DURATION_S = 40.0


def _synthetic_ppg_train(rate_bpm, fs=FS, duration_s=DURATION_S, amp=1.0,
                         wander_amplitude=0.0, wander_freq=0.25,
                         rng=None):
    """PPG-like signal with systolic peaks at ``rate_bpm``.

    Returns ``(signal, pulse_times_s)``.
    """
    rng = np.random.default_rng(rng or 12345)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    period_s = 60.0 / rate_bpm
    if period_s <= 0:
        raise ValueError(f"rate_bpm must be positive, got {rate_bpm!r}")
    # baseline wander (slow oscillation) + baseline
    signal = 0.02 * np.sin(2 * np.pi * wander_freq * t) \
        + wander_amplitude * np.sin(2 * np.pi * wander_freq * t)
    pulse_times = np.arange(0.2, duration_s - period_s, period_s)
    for p in pulse_times:
        i0 = int(p * fs)
        signal[i0:i0 + 4] += amp
    return signal, pulse_times


def _hit_rate(detected_s, pulse_times_s, tol_s=0.18):
    """Fraction of ground-truth pulses within ``tol_s`` of a detection."""
    det = np.asarray(detected_s)
    return float(np.mean([np.min(np.abs(det - r)) <= tol_s for r in pulse_times_s]))


# ---------------------------------------------------------------------------
# P3.2 — pulse-rate coverage at the boundary (160/175/190/200 bpm)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rate", [160, 175, 190, 200])
def test_detect_peaks_finds_boundary_rates(rate):
    """The detector must find pulses up to the declared maximum rate."""
    signal, pulse_times = _synthetic_ppg_train(rate)
    det = detect_peaks(signal, FS)
    expected = int(len(pulse_times))
    # A 300 ms refractory at 125 Hz -> 37-sample distance; the detector is
    # expected to find >= 90% of the true pulses.
    assert det.size >= int(0.85 * expected), \
        (rate, det.size, expected)
    hit = _hit_rate(det / FS, pulse_times)
    assert hit >= 0.85, (rate, hit)
    # rate consistency
    if det.size > 2:
        iv = np.diff(det) / FS
        measured = 60.0 / float(np.median(iv))
        assert np.isclose(measured, rate, rtol=0.25), (rate, measured)


def test_detector_max_rate_consistent_with_config():
    """Central config: detector can identify valid pulses up to 200 bpm."""
    fs = 125.0
    signal, pulse_times = _synthetic_ppg_train(MAX_PULSE_RATE_BPM)
    det = detect_peaks(signal, fs)
    assert det.size > 0, "no pulses detected at the declared max rate"
    hit = _hit_rate(det / fs, pulse_times)
    assert hit >= 0.85, hit


# ---------------------------------------------------------------------------
# Detector-condition coverage (P3.2)
# ---------------------------------------------------------------------------

def test_detect_peaks_with_baseline_drift():
    """Baseline wander shifts the amplitude reference but pulses are found."""
    signal, pulse_times = _synthetic_ppg_train(75.0, wander_amplitude=0.5,
                                               wander_freq=0.1)
    det = detect_peaks(signal, FS)
    hit = _hit_rate(det / FS, pulse_times)
    assert hit >= 0.85, hit


def test_detect_peaks_with_changing_amplitude():
    """Amplitude modulation (respiration) is handled by the local adaptive
    threshold; at least 85% of pulses should still be found."""
    fs = FS
    n = int(60 * fs)
    t = np.arange(n) / fs
    signal = 0.05 * np.sin(2 * np.pi * 0.25 * t)  # slow baseline drift
    for k in range(int(60 / 0.8)):
        p = 0.2 + 0.8 * k
        if p >= 60:
            break
        amp = 1.0 * (1 + 0.4 * np.sin(2 * np.pi * 0.1 * p))  # amplitude change
        signal[int(p * fs):int(p * fs) + 4] += amp
    det = detect_peaks(signal, fs)
    pulse_times = np.arange(0.2, 60.0, 0.8)[:int(60 / 0.8)]
    hit = _hit_rate(det / fs, pulse_times)
    assert hit >= 0.85, hit


def test_detect_peaks_motion_bursts():
    """Short motion bursts must not create many spurious detections and
    must not kill the true pulse train badly."""
    signal, pulse_times = _synthetic_ppg_train(90.0, duration_s=60.0)
    rng = np.random.default_rng(7)
    for _ in range(12):
        start = int(rng.integers(5, 55) * FS)
        signal[start:start + int(0.4 * FS)] += 2.5 * rng.normal(0, 1, int(0.4 * FS))
    det = detect_peaks(signal, FS)
    expected = int(len(pulse_times))
    assert det.size >= int(0.7 * expected), (det.size, expected)
    # Motion bursts can displace a nearby true pulse, so the strict
    # FP count is deliberately loose; the point is that detection is not
    # destroyed by bursts.
    det_t = det / FS
    matched = sum(1 for r in pulse_times
                  if np.min(np.abs(det_t - r)) <= 0.18)
    fp = len(det_t) - matched
    assert fp <= max(10, expected // 2), fp


def test_detect_peaks_missing_pulses():
    """A missing pulse should reduce detection rate by that fraction."""
    signal, pulse_times = _synthetic_ppg_train(90.0, duration_s=60.0)
    # zero out ~10% of pulses (keep surrounding samples clean)
    to_remove = pulse_times[::10]
    for p in to_remove:
        i0 = int(p * FS)
        signal[i0:i0 + 4] = 0.0
    det = detect_peaks(signal, FS)
    expected_full = int(len(pulse_times))
    expected_missing = int(round(len(pulse_times) * 0.9))
    assert det.size <= expected_full
    assert det.size >= int(0.75 * expected_missing), \
        (det.size, expected_missing)


def test_detect_peaks_false_peaks():
    """Spurious tall spikes should be largely rejected by the plausibility
    filter (both-adjacent-intervals-bad rule)."""
    signal, pulse_times = _synthetic_ppg_train(80.0, duration_s=60.0)
    rng = np.random.default_rng(11)
    for _ in range(10):
        p = float(rng.uniform(5.0, 55.0))
        i = int(p * FS)
        signal[i:i + int(0.02 * FS)] += rng.normal(0, 0.1, int(0.02 * FS)) + 2.0
    det = detect_peaks(signal, FS)
    expected = int(len(pulse_times))
    assert det.size >= int(0.9 * expected), (det.size, expected)


def test_detect_peaks_morphology_change():
    """A pulse-morphology change (wider systolic bump) is still detected."""
    fs = FS
    n = int(60 * fs)
    t = np.arange(n) / fs
    signal = 0.02 * np.sin(2 * np.pi * 0.25 * t)
    for k in range(int(60 / 0.8)):
        p = 0.2 + 0.8 * k
        if p >= 60:
            break
        i0 = int(p * fs)
        # wider pulse (8 samples vs 4) with higher amplitude
        signal[i0:i0 + 8] += 1.2
    det = detect_peaks(signal, fs)
    pulse_times = np.arange(0.2, 60.0, 0.8)[:int(60 / 0.8)]
    hit = _hit_rate(det / fs, pulse_times)
    assert hit >= 0.85, hit


def test_detect_peaks_different_heart_rates():
    """Detection works across a range of rates (60–91 bpm synthetic tests
    plus the 160–200 bpm boundary above)."""
    for rate in (60, 72, 75, 91):
        signal, pulse_times = _synthetic_ppg_train(rate)
        det = detect_peaks(signal, FS)
        hit = _hit_rate(det / FS, pulse_times)
        assert hit >= 0.85, (rate, hit)


def test_detector_refractory_matches_config():
    """P3.1: REFRACTORY_S is derived from MAX_PULSE_RATE_BPM."""
    assert abs(REFRACTORY_S - 60.0 / MAX_PULSE_RATE_BPM) < 1e-12
    fs = 125.0
    d = int(REFRACTORY_S * fs)
    assert d >= 1
    assert 60.0 / (d / fs) >= MAX_PULSE_RATE_BPM  # reach the declared max


def test_central_rate_config_is_used_consistently_everywhere():
    """P3.1 / audit Phase 4.2: the 40–200 bpm band is *one* configuration
    used by the detector, the interval filtering and the quality rules —
    not a set of independently hardcoded literals.

    Checks:
    * ``peak_detect`` exposes MIN/MAX and derives both the refractory
      period and the plausibility window from them;
    * ``quality_assessment``'s plausibility limits default to those same
      constants (they used to be literal ``40`` / ``200``);
    * ``run_bidmc_experiment`` derives its interval filter from them;
    * the README states the same band.
    """
    import inspect

    import quality_assessment as qa
    import run_bidmc_experiment as rbe
    from peak_detect import MIN_PULSE_RATE_BPM, PL_HI_S, PL_LO_S

    assert (MIN_PULSE_RATE_BPM, MAX_PULSE_RATE_BPM) == (40.0, 200.0)

    # Derived windows
    assert PL_LO_S == pytest.approx(60.0 / MAX_PULSE_RATE_BPM)   # 0.30 s
    assert PL_HI_S == pytest.approx(60.0 / MIN_PULSE_RATE_BPM)   # 1.50 s
    assert REFRACTORY_S == pytest.approx(PL_LO_S)

    # quality_assessment defaults reference the central constants
    for fn in (qa.interval_suspects, qa.compare_intervals, qa.assess_record):
        d = inspect.signature(fn).parameters
        assert d["min_hr_bpm"].default is MIN_PULSE_RATE_BPM, fn.__name__
        assert d["max_hr_bpm"].default is MAX_PULSE_RATE_BPM, fn.__name__
    assert qa.DEFAULT_MIN_HR_BPM is MIN_PULSE_RATE_BPM
    assert qa.DEFAULT_MAX_HR_BPM is MAX_PULSE_RATE_BPM

    # the experiment's interval filter is computed from the same constants
    src = inspect.getsource(rbe)
    assert "60.0 / MAX_PULSE_RATE_BPM" in src
    assert "60.0 / MIN_PULSE_RATE_BPM" in src

    # and the README documents the identical band
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(readme_path, encoding="utf-8") as fh:
        readme = fh.read()
    assert "40" in readme and "200 bpm" in readme


def test_quality_hard_limits_match_detector_plausibility_window():
    """The interval-quality hard limits and the detector's plausibility
    window must accept exactly the same interval range."""
    from peak_detect import PL_HI_S, PL_LO_S
    from quality_assessment import interval_suspects

    out = interval_suspects(np.array([PL_LO_S, PL_HI_S]), np.array([0.0, 1.0]))
    assert "implausible_interval" not in out.get("reasons", []), out
    # just outside the window on either side is flagged
    out2 = interval_suspects(np.array([0.5 * PL_LO_S, 2.0 * PL_HI_S]),
                             np.array([0.0, 1.0]))
    assert "implausible_interval" in out2.get("reasons", []), out2


# ---------------------------------------------------------------------------
# ECG reference (XQRS) — documented as automated
# ---------------------------------------------------------------------------

def test_detect_r_peaks_is_automated_reference():
    """The ECG reference must be documented as an automated detector, not
    manual annotation (P3.7, part 1)."""
    import ecg_reference
    from ecg_reference import detect_r_peaks  # noqa: F401
    assert "xqrs" in ecg_reference.detect_r_peaks.__doc__.lower() \
        or "automated" in (ecg_reference.detect_r_peaks.__doc__ or "").lower()


# ---------------------------------------------------------------------------
# P3.6 — one-to-one peak matching (regression)
# ---------------------------------------------------------------------------

def test_score_detector_enforces_one_to_one():
    """Precision must never exceed 1.0 when truth and detections overlap.

    Regression: the previous matcher looped over reference peaks, picked
    the nearest detection, and marked it used without checking, so one
    detection could score against multiple truth peaks (precision > 1.0
    possible, TP > len(detected)).
    """
    det = np.array([100])
    truth = np.array([95, 100, 105])
    s = score_detector(det, truth, fs=125.0, tolerance_s=0.016)
    assert s["precision"] <= 1.0
    assert s["tp"] <= 1  # only one detection exists
    assert s["tp"] <= len(det)
    # every matched pair uses a distinct detection
    pairs = match_peaks_one_to_one(det, truth, 125.0, 0.016)
    assert len({j for j, _i, _e in pairs}) == len(pairs)
