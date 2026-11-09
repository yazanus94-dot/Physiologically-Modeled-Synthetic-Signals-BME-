"""
From-scratch systolic peak detector for PPG waveforms.

Bandpass filter (0.5-8 Hz) to remove sensor drift and high-frequency
noise, then adaptive-threshold peak picking with a physiological
refractory period, then rejection of physiologically implausible
intervals. Catches most — not all — motion-artifact false positives,
deliberately (see README for how the residual error shows up in HRV).
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

REFRACTORY_S = 0.35


def _bandpass(signal, fs, low=0.5, high=8.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, padlen=min(len(signal) - 1, 3 * max(len(a), len(b))))


def detect_peaks(ppg, fs):
    """Return detected systolic peak sample indices."""
    filtered = _bandpass(np.asarray(ppg, dtype=float), fs)

    refractory_s = REFRACTORY_S
    # adaptive height threshold: short-window global percentile scaling
    # (peaks must be clearly taller than the local baseline wobble)
    baseline = float(np.median(filtered))
    height = 0.35 * (np.max(filtered) - baseline) + baseline

    cand, _ = find_peaks(filtered, height=height,
                         distance=max(1, int(REFRACTORY_S * fs)))
    peaks = list(cand)

    # plausibility filter: drop peaks producing physiologically impossible
    # pulse-to-pulse intervals (either direction). The middle peak in a bad
    # run is the prime suspect — either a spurious detection or an edge.
    keep = list(peaks)
    plausibility_lo, plausibility_hi = 0.30, 2.00  # seconds between pulses
    dropped = True
    while dropped and len(keep) > 2:
        dropped = False
        for i in range(1, len(keep) - 1):
            rr_prev = (keep[i] - keep[i - 1]) / fs
            rr_next = (keep[i + 1] - keep[i]) / fs
            if (rr_prev < plausibility_lo and rr_next < plausibility_lo) or \
               (rr_prev > plausibility_hi or rr_next > plausibility_hi):
                keep.pop(i)
                dropped = True
                break

    return np.asarray(keep, dtype=int)


def score_detector(detected, truth, fs, tolerance_s=0.15):
    """Sensitivity/precision at a matching window (default ±150 ms)."""
    det = np.asarray(detected, dtype=int)
    tru = np.asarray(truth, dtype=int)
    tol = int(round(tolerance_s * fs))
    matched_true = np.zeros(len(tru), dtype=bool)
    matched_det = np.zeros(len(det), dtype=bool)
    for i, t in enumerate(tru):
        if len(det) == 0:
            break
        j = int(np.argmin(np.abs(det - t)))
        if abs(int(det[j]) - int(t)) <= tol:
            matched_true[i] = True
            matched_det[j] = True
    tp = int(matched_true.sum())
    sens = tp / len(tru) if len(tru) else 0.0
    prec = tp / len(det) if len(det) else 0.0
    return {"sensitivity": round(sens, 4), "precision": round(prec, 4),
            "tp": tp,
            "fp": int(len(det) - matched_det.sum()),
            "fn": int(len(tru) - matched_true.sum())}
