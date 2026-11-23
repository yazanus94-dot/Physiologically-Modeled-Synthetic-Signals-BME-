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
  input SNR is the mixer's power ratio; the reported **improvement** is
  the reduction in *reconstruction-error* power (``10*log10(P_signal /
  P_error)`` before vs after), after zero-meaning.  That is the same
  error-based convention Project 2 uses, and it is why attenuating the
  input cannot masquerade as denoising (audit 2.5).
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

    Kept as a documented helper (and as Project 2's primary band limiter),
    but **not** the stage the reference submits: see
    :func:`lowpass_filter` for why the 0.5 Hz high-pass corner was removed
    from the benchmark pipeline after the audit measurement.
    """
    x = np.asarray(x, dtype=float)
    orig_n = x.size
    x = _pad_to_min(x)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)[:orig_n]


def lowpass_filter(x, fs, high=40.0, order=3):
    """Butterworth zero-phase low-pass at ``high`` Hz (default 40 Hz).

    Audit 2.5: the benchmark's clean reference is the **raw** MIT-BIH
    record (recorded with a 0.1 Hz hardware high-pass), so it still
    contains real low-frequency ECG content — P/T-wave bodies and the
    record's own residual baseline.  Applying a 0.5 Hz zero-phase
    high-pass to a 10 s window deletes part of that content and pays the
    ``sosfiltfilt`` transient cost over a 2-cycle window.  Measured on the
    360 real cells with the *correct* (error-based) SNR, running the old
    ``median → notch → 0.5–40 Hz band-pass`` chain on the **clean** signal
    (the best any denoiser using that chain could do) reaches only
    9.03 dB, whereas the same chain ending in this 40 Hz low-pass reaches
    11.38 dB.  The high-pass corner was therefore removing signal, not
    noise — the median stage already handles baseline wander.
    """
    x = np.asarray(x, dtype=float)
    orig_n = x.size
    x = _pad_to_min(x)
    sos = butter(order, high, btype="low", fs=fs, output="sos")
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


def combined_pipeline(x, fs, mains_hz=(50.0, 60.0)):
    """The clinical-style chain: median baseline removal → mains notches →
    40 Hz low-pass.

    ``mains_hz`` defaults to both standard mains frequencies; a notch is
    skipped when it would sit at or above Nyquist (e.g. 60 Hz on a 100 Hz
    record), so the chain stays valid for any sampling rate.
    """
    step = median_filter_baseline_removal(x, fs)
    for f0 in mains_hz:
        if 0.0 < f0 < 0.5 * fs:
            step = notch_filter(step, fs, freq=f0)
    return lowpass_filter(step, fs)


# ---------------------------------------------------------------------------
# Benchmark solution
# ---------------------------------------------------------------------------

@dataclass
class DenoisingResult:
    signal: np.ndarray
    method: str


def denoise(noisy, fs):
    """The reference pipeline: median baseline removal → 50/60 Hz notches →
    40 Hz low-pass (``combined_pipeline``).

    Why this chain, measured on the fixed evaluation protocol
    (``evaluator.evaluate(..., segments=10)`` on real MIT-BIH 118/119 +
    NSTDB ``bw``/``ma``/``em`` — 360 cells, error-based SNR):

    * **Median baseline removal (0.6 s)** attacks the baseline-wander
      class directly.  A linear 0.5 Hz high-pass also removes wander, but
      it additionally deletes low-frequency content that belongs to the
      clean record: measured on the 360 cells with the correct SNR, the
      chain ending in a 0.5–40 Hz band-pass reaches only **9.03 dB**
      fidelity when applied to the *clean* signal, versus **11.38 dB**
      when the same chain ends in the 40 Hz low-pass.  The high-pass was
      removing signal, not noise (audit 2.5).
    * **50 Hz and 60 Hz IIR notches** remove mains wherever it appears;
      they are what earns the signal-processing sub-score, which probes
      both standard frequencies.
    * **40 Hz low-pass** removes muscle/electrode high-frequency noise and
      keeps the diagnostic ECG band intact.

    Result (real data, 360 cells): mean SNR improvement **+3.89 dB** vs
    **+2.99 dB** for the band-pass-only starter (gap **+0.90 dB**), and
    an overall score of **0.8366** vs **0.6469** (gap **+0.19**).  The
    reference is better at every target SNR from −6 dB to +24 dB — see
    ``expected_results.json``.
    """
    return combined_pipeline(noisy, fs)


METHOD_NAME = "median+notch+lowpass_40hz"


def reference_solution():
    """Factory matching the evaluator's ``solution`` callable."""
    return _ReferenceSolution()


class _ReferenceSolution:
    """Callable that denoises a segment, mirroring ``DenoisingResult``."""

    def denoise(self, noisy, fs):
        return DenoisingResult(signal=denoise(noisy, fs), method=METHOD_NAME)
