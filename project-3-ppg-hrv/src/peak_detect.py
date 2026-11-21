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


def _local_height_threshold(filtered, fs, win_s=10.0, frac=0.35):
    """Per-sample adaptive height threshold from *local* signal statistics.

    A single global reference (one median + one 98th percentile for the
    whole record) assumes the pulse amplitude is stationary.  Real PPG is
    not: respiration, probe contact and slow drift move both the baseline
    and the systolic peak height over the course of a recording, so a
    threshold set from the record average is too high in the quiet spans
    and too low in the loud ones.

    This computes a rolling median (local baseline) and a rolling 98th
    percentile (local systolic peak height) on a coarse grid spaced half a
    window apart, then interpolates both to every sample and forms

        height[i] = base[i] + frac * (ref[i] - base[i])

    Returns ``None`` when the record is too short to estimate local
    statistics, so the caller can fall back to the global threshold.
    """
    x = np.asarray(filtered, dtype=float)
    n = int(x.size)
    win = max(int(round(float(win_s) * float(fs))), 1)
    if n < 2 * win:
        return None

    step = max(win // 2, 1)
    centers, bases, refs = [], [], []
    for start in range(0, n, step):
        seg = x[start:start + win]
        if seg.size < 8:
            continue
        b = float(np.median(seg))
        r = float(np.percentile(seg, 98))
        if not np.isfinite(b) or not np.isfinite(r):
            continue
        centers.append(start + seg.size / 2.0)
        bases.append(b)
        refs.append(r)

    if len(centers) < 2:
        return None

    centers = np.asarray(centers, dtype=float)
    bases = np.asarray(bases, dtype=float)
    refs = np.asarray(refs, dtype=float)

    # A local window can be flat or noise-dominated, giving ref <= base.
    # Fall back to the record-wide statistics for those spans rather than
    # emitting a threshold that would accept every sample.
    glob_b = float(np.median(x))
    glob_r = float(np.percentile(x, 98))
    if not (glob_r > glob_b):
        glob_r = float(np.max(x))
    bad = refs <= bases
    if np.any(bad):
        bases = np.where(bad, glob_b, bases)
        refs = np.where(bad, glob_r, refs)

    idx = np.arange(n, dtype=float)
    base = np.interp(idx, centers, bases)
    ref = np.interp(idx, centers, refs)
    height = base + float(frac) * (ref - base)
    return height


def detect_peaks(ppg, fs, local_adaptive=True):
    """Return detected systolic peak sample indices.

    ``local_adaptive`` (default) uses a per-sample rolling threshold that
    tracks slow changes in pulse amplitude; set it to ``False`` to use the
    single record-wide threshold (the original behaviour), which is kept
    so the two can be compared directly.
    """
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

    if local_adaptive:
        local = _local_height_threshold(filtered, fs)
        if local is not None:
            height = local

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


def match_peaks_one_to_one(detected, truth, fs, tolerance_s=0.15):
    """Greedy closest-first one-to-one peak matching.

    Returns a list of ``(detected_idx, truth_idx, error_ms)`` pairs.

    One-to-one is enforced on BOTH sides: a detected peak may be consumed
    by at most one reference peak and vice versa.  The previous
    implementation looped over reference peaks, picked the nearest
    detected peak, and never checked whether that detected peak had
    already been claimed.  Two reference peaks inside the same tolerance
    window could therefore both be scored against a single detection,
    which inflates TP above ``len(detected)`` and lets precision exceed
    1.0 (reproducible with ``detected=[100], truth=[95, 100, 105]``).

    Candidates are consumed closest-pair first, with index tie-breaks so
    the result is deterministic.
    """
    det = np.asarray(detected, dtype=np.int64).ravel()
    tru = np.asarray(truth, dtype=np.int64).ravel()
    tol = int(round(float(tolerance_s) * float(fs)))

    if det.size == 0 or tru.size == 0 or tol < 0:
        return []

    candidates = []
    for i, t in enumerate(tru):
        diffs = np.abs(det - t)
        for j in np.nonzero(diffs <= tol)[0]:
            candidates.append((int(diffs[j]), int(j), int(i)))

    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    used_det = set()
    used_true = set()
    pairs = []
    for dist, j, i in candidates:
        if j in used_det or i in used_true:
            continue
        used_det.add(j)
        used_true.add(i)
        error_ms = (int(det[j]) - int(tru[i])) * 1000.0 / float(fs)
        pairs.append((j, i, error_ms))

    pairs.sort(key=lambda p: p[1])   # ordered by reference peak
    return pairs


def score_detector(detected, truth, fs, tolerance_s=0.15):
    """Sensitivity/precision at a matching window (default ±150 ms).

    Matching is strictly one-to-one (see ``match_peaks_one_to_one``), so
    ``tp <= min(len(detected), len(truth))`` and precision ≤ 1.0.
    """
    pairs = match_peaks_one_to_one(detected, truth, fs, tolerance_s)
    n_det = int(np.asarray(detected, dtype=int).size)
    n_true = int(np.asarray(truth, dtype=int).size)
    tp = len(pairs)
    fp = n_det - tp
    fn = n_true - tp
    sens = tp / n_true if n_true else 0.0
    prec = tp / n_det if n_det else 0.0
    return {"sensitivity": round(sens, 4), "precision": round(prec, 4),
            "tp": tp, "fp": fp, "fn": fn}
