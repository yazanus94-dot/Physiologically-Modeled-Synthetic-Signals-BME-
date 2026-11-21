"""Fast unit tests for the generator and denoising methods."""
import os
import sys

import numpy as np

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
