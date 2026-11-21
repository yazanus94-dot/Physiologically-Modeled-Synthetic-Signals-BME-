"""
Four denoising approaches compared on the same noisy ECG signal:

  1. Median-filter baseline removal (classic clinical-ECG baseline wander
     correction: estimate baseline with a wide median filter, subtract it)
  2. Butterworth bandpass filtering (standard 0.5-40 Hz ECG passband)
  3. IIR notch filter targeted at the powerline frequency (50 Hz or 60 Hz,
     configurable — both mains standards are supported)
  4. FFT-domain soft-thresholding denoiser (frequency-domain analogue of
     wavelet soft-thresholding denoising, implemented without needing an
     external wavelet library)

And a combined pipeline (1 -> 3 -> 2) representing what a real clinical
ECG front-end preprocessing chain typically looks like.

Implementation notes (real-data upgrade)
----------------------------------------
* All IIR filtering is done with **second-order sections**
  (``scipy.signal.butter(..., output="sos")`` + ``sosfiltfilt``), which is
  numerically far more stable than applying ``filtfilt`` to a single
  high-order ``(b, a)`` transfer function — important because the real
  PhysioNet records run for many minutes at 360 Hz.
* All frequency parameters are validated (positive Nyquist, low < high,
  notch frequency strictly inside (0, Nyquist)) and segments too short for
  zero-phase (``sosfiltfilt``) padding raise a clear ``ValueError``
  instead of silently producing garbage.
* The public function signatures are unchanged from the original
  synthetic pipeline, so existing callers/tests keep working.
"""
import numpy as np
from scipy.signal import butter, iirnotch, medfilt, sosfiltfilt, tf2sos

# Powerline interference: the notch frequency is configurable so the same
# pipeline works for 50 Hz (Europe/Asia) and 60 Hz (Americas) mains.
DEFAULT_POWERLINE_HZ = 50.0
POWERLINE_FREQUENCIES_HZ = (50.0, 60.0)

# Below this many samples zero-phase filtering / median filtering cannot
# be applied meaningfully.
MIN_SIGNAL_LENGTH = 16


def _check_fs(fs):
    if not np.isscalar(fs) or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"sampling frequency must be a positive finite "
                         f"number, got {fs!r}")


def _check_band(low, high, fs):
    _check_fs(fs)
    nyq = 0.5 * fs
    if not (0 < low < high < nyq):
        raise ValueError(
            f"invalid frequency band: require 0 < low < high < Nyquist "
            f"({nyq:g} Hz), got low={low}, high={high}, fs={fs}")


def _sos_padlen(sos):
    """Number of samples ``sosfiltfilt`` will consume as edge padding
    (same formula scipy uses for its default ``padlen``)."""
    ntaps = 2 * sos.shape[0] + 1
    ntaps -= min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return 3 * ntaps


def _check_segment(signal, padlen, name):
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D signal, got shape {x.shape}")
    if x.size <= max(padlen, 0):
        raise ValueError(
            f"{name}: segment too short ({x.size} samples); zero-phase "
            f"filtering needs more than {padlen} samples")
    return x


def median_filter_baseline_removal(signal, fs, window_s=0.6):
    _check_fs(fs)
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s!r}")
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("expected a non-empty 1-D signal")
    window = int(round(window_s * fs))
    if window % 2 == 0:
        window += 1
    window = max(window, 3)
    if window > x.size:
        raise ValueError(
            f"median window ({window} samples) is longer than the segment "
            f"({x.size} samples)")
    baseline = medfilt(x, kernel_size=window)
    return x - baseline, baseline


def bandpass_filter(signal, fs, low=0.5, high=40.0, order=3):
    _check_band(low, high, fs)
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}")
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    x = _check_segment(signal, _sos_padlen(sos), "bandpass_filter")
    return sosfiltfilt(sos, x)


def notch_filter(signal, fs, freq=DEFAULT_POWERLINE_HZ, quality=30.0):
    _check_fs(fs)
    nyq = 0.5 * fs
    if not np.isfinite(freq) or not (0 < freq < nyq):
        raise ValueError(
            f"notch frequency must satisfy 0 < freq < Nyquist ({nyq:g} Hz), "
            f"got {freq} (fs={fs})")
    if not np.isfinite(quality) or quality <= 0:
        raise ValueError(f"quality factor must be positive, got {quality!r}")
    b, a = iirnotch(freq, quality, fs=fs)
    sos = tf2sos(b, a)
    x = _check_segment(signal, _sos_padlen(sos), "notch_filter")
    return sosfiltfilt(sos, x)


def fft_soft_threshold_denoise(signal, fs, keep_band=(0.5, 40.0),
                               threshold_fraction=0.08):
    _check_fs(fs)
    low, high = keep_band
    _check_band(low, high, fs)
    if not np.isfinite(threshold_fraction) or not (0 <= threshold_fraction <= 1):
        raise ValueError(
            f"threshold_fraction must be in [0, 1], got {threshold_fraction}")
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("expected a 1-D signal with at least 2 samples")
    n = x.size
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    out_of_band = (freqs < low) | (freqs > high)
    spectrum[out_of_band] = 0.0

    mag = np.abs(spectrum)
    if mag.max() == 0:
        return np.fft.irfft(spectrum, n=n)
    thresh = threshold_fraction * mag.max()
    magnitude_reduced = np.maximum(mag - thresh, 0)
    phase = np.angle(spectrum)
    spectrum_denoised = magnitude_reduced * np.exp(1j * phase)

    return np.fft.irfft(spectrum_denoised, n=n)


def combined_pipeline(signal, fs, powerline_hz=DEFAULT_POWERLINE_HZ):
    # powerline_hz is passed to notch_filter(), which validates that
    # 0 < powerline_hz < Nyquist. Both standard mains frequencies
    # (50 Hz and 60 Hz) are supported; see POWERLINE_FREQUENCIES_HZ.
    step1, baseline = median_filter_baseline_removal(signal, fs)
    step2 = notch_filter(step1, fs, freq=powerline_hz)
    step3 = bandpass_filter(step2, fs)
    return step3


def snr_db(clean, estimate):
    clean = np.asarray(clean, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if clean.shape != estimate.shape:
        raise ValueError("clean and estimate must have the same shape")
    noise = clean - estimate
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def rmse(clean, estimate):
    clean = np.asarray(clean, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if clean.shape != estimate.shape:
        raise ValueError("clean and estimate must have the same shape")
    return float(np.sqrt(np.mean((clean - estimate) ** 2)))
