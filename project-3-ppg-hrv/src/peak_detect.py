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
    # Adaptive height threshold.  The reference amplitude must be robust to
    # artefacts: a single motion spike inflates ``np.max`` and the resulting
    # threshold then rejects every beat of normal amplitude (measured on real
    # BIDMC data: max/p98 = 1.6-2.1, and the max-based threshold dropped 55
    # of 727 beats in bidmc02 — recall 91.9% vs 99.3% for a p98 reference,
    # at identical ~99% precision).  The 98th percentile tracks the true
    # systolic peak height while ignoring the top 2% of samples.
    baseline = float(np.median(filtered))
    reference = float(np.percentile(filtered, 98))
    if reference <= baseline:          # degenerate/flat signal
        reference = float(np.max(filtered))
    height = 0.35 * (reference - baseline) + baseline

    cand, _ = find_peaks(filtered, height=height,
                         distance=max(1, int(REFRACTORY_S * fs)))
    peaks = list(cand)

    # Plausibility filter: drop peaks producing physiologically impossible
    # pulse-to-pulse intervals.  A peak is removed only when BOTH of its
    # adjacent intervals are implausible — i.e. it is a double detection
    # (both intervals < 0.30 s) or an isolated spike sitting between two
    # long gaps (both intervals > 2.0 s).
    #
    # The previous rule dropped a peak when EITHER adjacent interval was
    # > 2.0 s.  Because each removal merges its two neighbouring intervals,
    # a single genuine signal dropout (one 7.9 s gap in a real BIDMC
    # record) made the merged interval even longer, which violated the rule
    # again for the next peak, and the cascade deleted every peak in the
    # record except two (measured: 707 candidates -> 2 kept).  Requiring
    # BOTH sides to be bad makes the filter self-limiting: after a removal
    # the neighbour sees the merged interval on one side only, so the chain
    # stops.  Long gaps themselves (missed beats / sensor dropout) are NOT
    # repaired by deleting a real peak — they are flagged downstream by
    # quality_assessment.interval_suspects and compare_intervals.
    plausibility_lo, plausibility_hi = 0.30, 2.00  # seconds between pulses
    keep = list(peaks)
    dropped = True
    while dropped and len(keep) > 2:
        dropped = False
        for i in range(1, len(keep) - 1):
            rr_prev = (keep[i] - keep[i - 1]) / fs
            rr_next = (keep[i + 1] - keep[i]) / fs
            prev_bad = rr_prev < plausibility_lo or rr_prev > plausibility_hi
            next_bad = rr_next < plausibility_lo or rr_next > plausibility_hi
            if prev_bad and next_bad:
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
