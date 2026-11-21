"""
Mixes real noise into real ECG at a precisely controlled SNR.

Method
------
1. **Resampling** — if the ECG and the noise were recorded at different
   sampling frequencies, both are resampled to a common frequency with
   ``scipy.signal.resample_poly`` (polyphase, aliasing-protected). The
   ratio is derived from the two rates via ``fractions.Fraction`` so no
   sampling frequency is hardcoded.
2. **Alignment** — after resampling, both arrays are truncated to the
   common (minimum) length, sample 0 of each being sample 0 of its own
   segment.
3. **Scaling** — the noise is scaled so that the *segment powers*
   satisfy the standard definition

       SNR_dB = 10 * log10( P_signal / P_noise ),
       P = mean(x**2)

   measured on the signal-only and noise-only segments. The gain is
   ``g = sqrt( P_s / (P_n * 10**(SNR/10)) )``; the achieved SNR is
   recomputed from the actual arrays that get mixed, so rounding in the
   resampler cannot silently shift the operating point.
4. **Mixing** — ``noisy = clean + g * noise``.

The six SNR operating points used by the experiment are the standard
NSTDB levels: ``TARGET_SNR_LEVELS = [-6, 0, 6, 12, 18, 24]`` dB.
"""
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

# NSTDB-style SNR operating points (dB), supported by every function in
# this module (any scalar target is also accepted).
TARGET_SNR_LEVELS = [-6, 0, 6, 12, 18, 24]


def signal_power(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("cannot compute power of an empty signal")
    return float(np.mean(x ** 2))


def resample_to(x, fs_from, fs_to):
    """Resample ``x`` from ``fs_from`` to ``fs_to`` with a polyphase
    FIR filter. Returns a copy when the rates are equal."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"expected a non-empty 1-D signal, got shape "
                         f"{x.shape}")
    if not (np.isfinite(fs_from) and fs_from > 0 and
            np.isfinite(fs_to) and fs_to > 0):
        raise ValueError(f"sampling rates must be positive and finite, "
                         f"got fs_from={fs_from!r}, fs_to={fs_to!r}")
    if float(fs_from) == float(fs_to):
        return x.copy()
    ratio = Fraction(float(fs_to) / float(fs_from)).limit_denominator(10000)
    if ratio.numerator < 1 or ratio.denominator < 1:
        raise ValueError(f"cannot resample from {fs_from} Hz to {fs_to} Hz")
    return resample_poly(x, ratio.numerator, ratio.denominator)


def align_to_common(signal, noise):
    """Truncate both arrays to their common (minimum) length."""
    n = min(len(signal), len(noise))
    if n == 0:
        raise ValueError("cannot align empty signals")
    return np.asarray(signal, dtype=float)[:n], \
        np.asarray(noise, dtype=float)[:n]


def scale_noise_to_snr(signal, noise, target_snr_db):
    """Scale ``noise`` so that 10*log10(P_signal/P_noise) equals
    ``target_snr_db``. Returns ``(scaled_noise, achieved_snr_db)``."""
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if signal.shape != noise.shape:
        raise ValueError("signal and noise must have the same shape to be "
                         f"mixed, got {signal.shape} vs {noise.shape}")
    target = float(target_snr_db)
    if not np.isfinite(target):
        raise ValueError(f"target_snr_db must be finite, got {target_snr_db!r}")
    p_signal = signal_power(signal)
    p_noise = signal_power(noise)
    if p_signal <= 0:
        raise ValueError("signal has zero power — nothing to mix")
    if p_noise <= 0:
        raise ValueError("noise has zero power — cannot reach a finite SNR")
    gain = np.sqrt(p_signal / (p_noise * 10.0 ** (target / 10.0)))
    scaled = noise * gain
    achieved = 10.0 * np.log10(p_signal / signal_power(scaled))
    return scaled, achieved


def mix_to_snr(clean, noise, target_snr_db, fs_clean=None, fs_noise=None,
               fs_out=None):
    """Resample (if needed), align, scale and mix.

    Parameters
    ----------
    clean : array_like — the clean reference signal
    noise : array_like — the noise segment
    target_snr_db : float — desired SNR, e.g. one of TARGET_SNR_LEVELS
    fs_clean, fs_noise : float, optional — sampling rates of the inputs;
        when they differ the noise is resampled to the ECG's rate
    fs_out : float, optional — force an output sampling rate

    Returns
    -------
    (noisy, achieved_snr_db)
    """
    fs_out = fs_out if fs_out is not None else \
        (fs_clean if fs_clean is not None else fs_noise)
    if fs_out is not None and not (np.isfinite(fs_out) and fs_out > 0):
        raise ValueError(f"fs_out must be positive and finite, got {fs_out!r}")

    clean = np.asarray(clean, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if fs_clean is not None and fs_out is not None and \
            float(fs_clean) != float(fs_out):
        clean = resample_to(clean, fs_clean, fs_out)
    if fs_noise is not None and fs_out is not None and \
            float(fs_noise) != float(fs_out):
        noise = resample_to(noise, fs_noise, fs_out)

    clean, noise = align_to_common(clean, noise)
    scaled, achieved = scale_noise_to_snr(clean, noise, target_snr_db)
    noisy = clean + scaled
    return noisy, achieved
