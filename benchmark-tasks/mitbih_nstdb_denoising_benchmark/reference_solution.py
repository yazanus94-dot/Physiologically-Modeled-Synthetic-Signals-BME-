"""Reference solution for the MIT-BIH + NSTDB ECG denoising benchmark.

This is a complete, documented implementation.  It mirrors the primary
denoising pipeline of ``project-2-ecg-denoising`` (Physiologically
Modeled Synthetic Signals repo) but is self-contained (numpy/scipy only)
so a candidate can run and read it standalone.

Scientific notes
----------------
- All waveform-error metrics use the same **DC-offset convention** as the
  reference implementation: signals are mean-removed before scoring
  (``_zero_mean``).  A legitimate high-pass that removes a DC offset
  should not be penalised for doing its job — this is what makes the
  metrics comparable across methods and SNRs.
- The benchmark mixes clean MIT-BIH ECG with real NSTDB noise at an
  **exactly requested target SNR** (mixer accuracy ~1e-15 dB).  The
  reported SNR improvement therefore uses the same power-ratio
  convention as the noise mixer, applied after zero-meaning.
- Filters are zero-phase (``sosfiltfilt``), matching the reference
  implementation, so denoising introduces no phase distortion.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, medfilt, sosfiltfilt, tf2sos

# ---------------------------------------------------------------------------
# Preprocessing primitives (same conventions as the repo, documented here)
# ---------------------------------------------------------------------------

def zero_mean(x):
    """Remove the DC level of a signal (see _zero_mean in evaluate_denoising.py)."""
    x = np.asarray(x, dtype=float)
    return x - x.mean()


MIN_SEGMENT_SAMPLES = 64   # scipy filtfilt padlen for the 3-stage Butterworth


def _pad_to_min(x, min_len=MIN_SEGMENT_SAMPLES):
    """Edge-pad a segment to ``min_len`` samples (so zero-phase filters do
    not raise on short input); the caller slices back afterwards."""
    n = int(np.asarray(x).size)
    if n >= min_len:
        return np.asarray(x, dtype=float)
    if n == 0:
        return np.zeros(min_len, dtype=float)
    return np.pad(np.asarray(x, dtype=float), (0, min_len - n),
                  mode="edge")


def bandpass_filter(x, fs, low=0.5, high=40.0, order=3):
    """Butterworth zero-phase band-pass 0.5–40 Hz (defaults as in repo).

    Handles very short input by edge-padding to the minimum length and
    slicing back to the original length.
    """
    x = np.asarray(x, dtype=float)
    orig_n = x.size
    x = _pad_to_min(x)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)[:orig_n]


def notch_filter(x, fs, freq=50.0, quality=30.0):
    """IIR notch at ``freq`` Hz with the standard ``Q`` factor."""
    x = np.asarray(x, dtype=float)
    orig_n = x.size
    x = _pad_to_min(x)
    b, a = iirnotch(freq, quality, fs=fs)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, x)[:orig_n]


def median_filter_baseline_removal(x, fs, window_s=0.6):
    """Subtract the median baseline computed with a 0.6 s sliding median.

    ``scipy.signal.medfilt`` needs an odd kernel that fits the input, so
    on segments shorter than the requested window the largest odd length
    that fits is used (this avoids scipy's "kernel_size exceeds volume
    extent" zero-padding warning on the evaluator's short-segment probe).
    """
    x = np.asarray(x, dtype=float)
    orig_n = x.size
    x = _pad_to_min(x)          # guarantees >= MIN_SEGMENT_SAMPLES (64)
    window = max(3, round(window_s * fs) | 1)
    if window > x.size:
        window = x.size if x.size % 2 == 1 else x.size - 1
    baseline = medfilt(x, kernel_size=window)
    return (x - baseline)[:orig_n]


def fft_soft_threshold(x, fs, keep_band=(0.5, 40.0), threshold_fraction=0.08):
    """FFT-domain soft-threshold denoiser (repo's ``fft_soft_threshold_denoise``).

    Out-of-band bins are zeroed, then every in-band bin's **magnitude** is
    soft-thresholded at ``threshold_fraction * max|magnitude|`` while its
    **phase is preserved** — the frequency-domain analogue of wavelet
    soft-thresholding.

    Audit fix: this helper used to compare an *amplitude* against a
    *total band power* (units mismatch, threshold ≫ every bin) and to
    subtract a real scalar from complex coefficients, which zeroed the
    whole spectrum and returned silence.  It now matches
    ``project-2-ecg-denoising/src/denoise_methods.py`` exactly and is
    covered by ``test_fft_soft_threshold_preserves_signal``.
    """
    x = np.asarray(x, dtype=float)
    low, high = keep_band
    n = x.size
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    in_band = (freqs >= low) & (freqs <= high)
    spectrum[~in_band] = 0.0

    mag = np.abs(spectrum)
    if mag.max() == 0:
        return np.fft.irfft(spectrum, n)
    thresh = threshold_fraction * mag.max()
    magnitude_reduced = np.maximum(mag - thresh, 0.0)
    phase = np.angle(spectrum)
    return np.fft.irfft(magnitude_reduced * np.exp(1j * phase), n)


def combined_pipeline(x, fs, powerline_hz=50.0):
    """The clinical-style chain: median baseline removal → notch → bandpass."""
    step1 = median_filter_baseline_removal(x, fs)
    step2 = notch_filter(step1, fs, freq=powerline_hz)
    return bandpass_filter(step2, fs)


# ---------------------------------------------------------------------------
# Benchmark solution
# ---------------------------------------------------------------------------

@dataclass
class DenoisingResult:
    signal: np.ndarray
    method: str


def denoise(noisy, fs):
    """The reference pipeline: median baseline removal → 50 Hz notch →
    0.5–40 Hz band-pass (``combined_pipeline``).

    Why this chain, measured on the fixed evaluation protocol
    (``evaluator.evaluate(..., segments=10)`` on real MIT-BIH 118/119 +
    NSTDB ``bw``/``ma``/``em``):

    * **Median baseline removal (0.6 s)** attacks the baseline-wander
      class directly; a linear high-pass would also remove it but spreads
      QRS energy and rings on the steepest NSTDB wander.
    * **50 Hz IIR notch** removes the mains component that sits just
      above the pass-band, where a 3rd-order band-pass still passes a
      measurable amount of it.
    * **0.5–40 Hz Butterworth band-pass** finalises the ECG band.

    Result: mean SNR improvement **+3.16 dB** vs **+2.65 dB** for the
    band-pass-only starter (measured gap **+0.51 dB**, and a strictly
    higher overall score at every target SNR from −6 dB to +24 dB —
    see ``expected_results.json``).
    """
    return combined_pipeline(noisy, fs, powerline_hz=50.0)


METHOD_NAME = "median+notch+bandpass_0.5_40hz"


def reference_solution():
    """Factory matching the evaluator's ``solution`` callable."""
    return _ReferenceSolution()


class _ReferenceSolution:
    """Callable that denoises a segment, mirroring ``DenoisingResult``."""

    def denoise(self, noisy, fs):
        return DenoisingResult(signal=denoise(noisy, fs), method=METHOD_NAME)
