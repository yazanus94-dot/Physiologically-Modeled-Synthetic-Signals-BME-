"""
Beat-level feature extraction for ECG arrhythmia classification.

Features per detected beat (classes are not used here — this is purely
signal-morphology based):
  RR-interval features  - pre-RR, post-RR, RR ratio  (captures prematurity
                          and the compensatory pause pattern)
  QRS morphology        - amplitude, width, energy  (captures the wide/
                          bizarre PVC shape vs the narrow normal QRS)
"""
import numpy as np

QRS_WINDOW_S = 0.12  # analysis window around the detected R peak


def extract_features(signal, r_peaks, fs):
    """Return (X, feature_names) but only where both neighbours exist."""
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 3:
        return np.empty((0, 6)), ["rr_prev", "rr_next", "rr_ratio",
                                  "qrs_amp", "qrs_width", "qrs_energy"]

    rr_intervals = np.diff(r_peaks) / fs  # seconds

    X = []
    for i, r in enumerate(r_peaks):
        if i == 0 or i == len(r_peaks) - 1:
            continue
        rr_prev = rr_intervals[i - 1]
        rr_next = rr_intervals[i]

        w = int(round(QRS_WINDOW_S * fs))
        lo = max(0, r - w)
        hi = min(len(signal), r + w)
        seg = signal[lo:hi]

        base = float(np.median(signal[max(0, r - w * 2):min(len(signal), r + w * 2)]))
        amp = float(np.max(np.abs(seg - base)))

        # QRS width: stay above 30% of peak absolute deviation from baseline
        above = np.abs(seg - base) > 0.30 * amp
        width = float(above.sum()) / fs

        # energy: sum of squared derivative in the window — captures the
        # sharp/wide trade-off between normal and PVC beats
        d = np.diff(seg)
        energy = float(np.sum(d * d)) / (w * w)

        X.append([rr_prev, rr_next, rr_next / max(rr_prev, 1e-9),
                  amp, width, energy])

    names = ["rr_prev", "rr_next", "rr_ratio", "qrs_amp", "qrs_width", "qrs_energy"]
    return np.asarray(X, dtype=float), names
