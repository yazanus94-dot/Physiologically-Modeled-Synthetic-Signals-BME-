"""Fast unit tests for the generator and denoising methods."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_noisy_signal import make_clean_and_noisy  # noqa: E402
from denoise_methods import (bandpass_filter, notch_filter,  # noqa: E402
                             combined_pipeline, snr_db, rmse)
from evaluate_denoising import _zero_mean  # noqa: E402


def test_generation_shapes():
    clean, noisy, fs, r_peaks, labels, comps = make_clean_and_noisy(
        duration_s=10, fs=250, seed=3)
    # chunk placement can run slightly past the target duration — the
    # generator is the user's original code, kept as-is
    assert len(clean) == len(noisy) >= 10 * 250
    assert fs == 250
    assert len(r_peaks) == len(labels)


def test_bandpass_improves_snr():
    clean, noisy, fs, *_ = make_clean_and_noisy(duration_s=10, fs=250, seed=3)
    assert snr_db(clean, noisy) < 0
    for est in [
        bandpass_filter(noisy, fs),
        notch_filter(noisy, fs),
        combined_pipeline(noisy, fs),
    ]:
        assert snr_db(clean, est) > snr_db(clean, noisy)
        assert rmse(clean, est) < rmse(clean, noisy)


# ---------------------------------------------------------------------------
# DC-offset scoring regression
# ---------------------------------------------------------------------------

def _waveform(n, fs):
    t = np.arange(n) / fs
    return (1.0 * np.sin(2 * np.pi * 1.1 * t)
            + 0.4 * np.sin(2 * np.pi * 3.3 * t + 0.7))


def test_pure_dc_shift_is_not_scored_as_error():
    """A perfect high-pass of a DC-offset signal must score as near-perfect.

    Real defect this locks down: the clean MIT-BIH segment used by the
    real experiment has mean ~ -0.33 mV.  Filters that remove DC were
    scored against the DC-inclusive reference, so every one of them capped
    out at ~1 dB output SNR no matter how clean the input — reported as a
    fictitious -23 dB 'denoising improvement' at 24 dB input SNR.
    """
    from evaluate_denoising import compute_segment_metrics, _zero_mean

    fs, n = 250, 30 * 250
    clean = _waveform(n, fs) - 0.33          # electrode/baseline offset
    estimate = _zero_mean(clean)             # exactly what a high-pass does

    m = compute_segment_metrics(clean, estimate, fs)

    assert m["snr_db"] > 30                  # not ~1 dB
    assert m["nrmse"] < 0.02
    assert m["rmse"] < 1e-9
    assert m["prd"] < 1e-6
    assert abs(m["pearson"] - 1.0) < 1e-9    # already offset-invariant


def test_dc_correction_does_not_hide_real_distortion():
    """Offset removal must not blind the metrics to genuine error."""
    from evaluate_denoising import compute_segment_metrics

    fs, n = 250, 30 * 250
    clean = _waveform(n, fs) - 0.33
    distorted = _zero_mean(clean) * 0.5      # real amplitude distortion

    m = compute_segment_metrics(clean, distorted, fs)
    assert m["snr_db"] < 10                  # still clearly penalised
    assert m["nrmse"] > 0.10                 # halving amplitude is caught


def test_zero_mean_removes_only_the_mean():
    from evaluate_denoising import _zero_mean

    x = np.array([1.0, 2.0, 3.0, 100.0])
    z = _zero_mean(x)
    assert abs(z.mean()) < 1e-12
    assert np.allclose(np.diff(z), np.diff(x))   # shape untouched


def test_metrics_still_penalise_a_delayed_estimate():
    """A time-shifted copy is real error and must stay penalised."""
    from evaluate_denoising import compute_segment_metrics

    fs, n = 250, 30 * 250
    clean = _waveform(n, fs) - 0.33
    delayed = np.roll(clean, 40)             # 160 ms shift

    m = compute_segment_metrics(clean, delayed, fs)
    assert m["snr_db"] < 20
    assert m["pearson"] < 0.99


# ---------------------------------------------------------------------------
# R-peak F1 (audit item: F1 in metrics)
# ---------------------------------------------------------------------------

def test_rpeak_f1_metric():
    """F1 = 2·P·S/(P + S), computed from the matched pairs."""
    from evaluate_denoising import rpeak_metrics
    # ref at 1000, 2000; det at 1005, 2005 -> both within ±38 samples
    # (150 ms at 250 Hz) -> tp = 2 -> S = P = 1 -> F1 = 1
    m = rpeak_metrics([1000, 2000], [1005, 2005], fs=250.0)
    assert m["rpeak_f1"] == pytest.approx(1.0)
    # ref at 1000, 2000; det at 1000, 3000 -> only ref 1000 matches
    # (ref 2000 is 1000 samples away > tolerance) -> tp = 1
    m2 = rpeak_metrics([1000, 2000], [1000, 3000], fs=250.0)
    assert m2["rpeak_tp"] == 1
    assert m2["rpeak_f1"] == pytest.approx(0.5)
    # no peaks at all -> NaN, not a divide-by-zero
    m3 = rpeak_metrics([], [], fs=250.0)
    assert np.isnan(m3["rpeak_f1"])


# ---------------------------------------------------------------------------
# NaN / infinity handling (audit item: NaN/infinity handling)
# ---------------------------------------------------------------------------

def test_compute_segment_metrics_rejects_nonfinite_inputs():
    """Fail fast with a clear error on NaN/inf rather than propagating
    NaN into CSV/JSON/figures."""
    from evaluate_denoising import compute_segment_metrics
    n = 3000
    clean = _waveform(n, 250)
    bad_nan = clean.copy(); bad_nan[1000] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_segment_metrics(bad_nan, clean, 250)
    bad_inf = clean.copy(); bad_inf[1000] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        compute_segment_metrics(bad_inf, clean, 250)
    # finite input still works
    m = compute_segment_metrics(clean, clean, 250)
    assert np.isfinite(m["rmse"])


def test_median_filter_output_finite_and_shaped():
    """median_filter_baseline_removal returns a finite, same-shape array."""
    from denoise_methods import median_filter_baseline_removal
    n = 3000
    clean = _waveform(n, 250)
    est = median_filter_baseline_removal(clean, 250)[0]
    assert est.shape == clean.shape
    assert np.all(np.isfinite(est))


# ---------------------------------------------------------------------------
# Filter edge effects (audit item: filter edge effects)
# ---------------------------------------------------------------------------

def test_filter_edge_effects():
    """Zero-phase filters (filtfilt) handle edges via reflection; outputs
    are finite and close to the input at the very start/end."""
    from denoise_methods import (bandpass_filter, notch_filter,
                                 median_filter_baseline_removal)
    fs, n = 250, 3000
    clean = _waveform(n, fs) + 0.1
    for fn in (bandpass_filter, notch_filter):
        est = fn(clean, fs)
        assert np.all(np.isfinite(est)), fn.__name__
        assert est.shape == clean.shape
        # within 25 ms of either end the filter output should track the
        # input (filtfilt is edge-reflecting; allow a loose bound)
        edge = max(1, int(0.025 * fs))
        assert np.max(np.abs(est[:edge] - clean[:edge])) < 1.0, fn.__name__
        assert np.max(np.abs(est[-edge:] - clean[-edge:])) < 1.0, fn.__name__
    est_m, _ = median_filter_baseline_removal(clean, fs)
    assert np.all(np.isfinite(est_m))
    assert est_m.shape == clean.shape


# ---------------------------------------------------------------------------
# Deterministic experiment generation (audit item)
# ---------------------------------------------------------------------------

def test_make_clean_and_noisy_is_deterministic():
    """Same seed -> byte-identical clean, noisy and component arrays."""
    from generate_noisy_signal import make_clean_and_noisy
    kwargs = dict(duration_s=10, fs=250, seed=3)
    c1, n1, *_ = make_clean_and_noisy(**kwargs)
    c2, n2, *_ = make_clean_and_noisy(**kwargs)
    assert np.array_equal(c1, c2)
    assert np.array_equal(n1, n2)
    # a different seed changes the output
    c3, _, *_ = make_clean_and_noisy(duration_s=10, fs=250, seed=4)
    assert not np.array_equal(c1, c3)
