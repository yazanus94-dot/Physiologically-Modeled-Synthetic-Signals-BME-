"""
Beat-level feature extraction for ECG arrhythmia classification.

Features per detected beat (classes are not used here — this is purely
signal-morphology based):

RR-interval features:
  - Previous RR interval (prematurity)
  - Following RR interval (compensatory pause)
  - Mean local RR (context)
  - RR ratio (prematurity index)
  - Prematurity ratio

QRS morphology features:
  - QRS amplitude
  - QRS width
  - QRS energy
  - Beat-window mean
  - Beat-window standard deviation
  - Beat-window variance
  - Skewness
  - Kurtosis

Optional waveform morphology features:
  - R-peak amplitude relative to baseline
  - Q-wave presence/amplitude
  - S-wave presence/amplitude
  - T-wave features (amplitude, slope)
"""
import numpy as np
from scipy.stats import skew, kurtosis

from typing import Optional, Tuple, List

# Default window around R-peak for QRS analysis
QRS_WINDOW_S = 0.12  # 120 ms total (±60 ms)


def extract_rr_features(r_peaks: np.ndarray, fs: float) -> Tuple[np.ndarray, list]:
    """
    Extract RR-interval features.

    Parameters
    ----------
    r_peaks : np.ndarray
        Sample indices of detected R-peaks
    fs : float
        Sampling frequency

    Returns
    -------
    features : np.ndarray
        Shape (n_beats - 2, 5) - features for beats with both neighbors
    names : list of str
        Feature names
    """
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 3:
        return np.empty((0, 5)), ["rr_prev", "rr_next", "rr_mean_local",
                                   "rr_ratio", "prematurity_ratio"]

    rr_intervals = np.diff(r_peaks) / fs  # seconds

    X = []
    for i in range(1, len(r_peaks) - 1):
        rr_prev = rr_intervals[i - 1]
        rr_next = rr_intervals[i]

        # Local mean RR (excluding current beat)
        local_rr = np.concatenate([rr_intervals[max(0, i - 5):i],
                                    rr_intervals[i + 1:min(len(rr_intervals), i + 6)]])
        rr_mean_local = np.mean(local_rr) if len(local_rr) > 0 else rr_prev

        # RR ratio (next/prev) - captures prematurity
        rr_ratio = rr_next / max(rr_prev, 1e-9)

        # Prematurity ratio: how much shorter is this RR vs local mean
        prematurity_ratio = rr_prev / max(rr_mean_local, 1e-9)

        X.append([rr_prev, rr_next, rr_mean_local, rr_ratio, prematurity_ratio])

    names = ["rr_prev", "rr_next", "rr_mean_local", "rr_ratio", "prematurity_ratio"]
    return np.asarray(X, dtype=float), names


def extract_rr_features_for_annotations(
    ann_samples: np.ndarray,
    ann_symbols: List[str],
    target_indices: np.ndarray,
    fs: float,
    local_window: int = 5,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    RR-interval features for an arbitrary subset of annotated beats.

    This is the annotation-centred counterpart of :func:`extract_rr_features`.
    It exists because the AAMI pipeline segments beats at *reference
    annotation* positions (and discards non-AAMI beats), so the neighbour
    structure has to be rebuilt from the full annotation series rather than
    from the retained beat windows.

    RR intervals are measured between consecutive **beat** symbols (see
    ``aami.BEAT_SYMBOLS``), which is wider than the four AAMI classes: a paced
    ('/') or unclassifiable ('Q') beat is not a class, but it is a real QRS
    and therefore still terminates the preceding RR interval. Non-beat
    annotations (rhythm markers, signal-noise flags) are ignored.

    Parameters
    ----------
    ann_samples : np.ndarray
        Sample index of every annotation in the record, in time order.
    ann_symbols : list of str
        Symbol of every annotation (same length/order as ``ann_samples``).
    target_indices : np.ndarray
        Indices into the annotation arrays of the beats to score, as returned
        by ``aami.segment_beats_annotation_centered``.
    fs : float
        Sampling frequency in Hz (read from the WFDB header, never hardcoded).
    local_window : int
        Half-width, in beats, of the local RR context window.

    Returns
    -------
    features : np.ndarray
        Shape ``(n_kept, 5)``. Rows are aligned with ``target_indices[keep]``.
    names : list of str
        Feature names, in order.
    keep_mask : np.ndarray of bool
        Shape ``(len(target_indices),)``. ``False`` where the beat has no
        preceding/following RR interval (record boundary), in which case the
        caller must drop the corresponding beat window and label.

    Notes
    -----
    No label information enters these features: they are computed purely from
    annotation *timing*, which is available at inference time for a real
    deployment only if a beat detector runs first. ``map_to_aami_class`` is
    never consulted here.
    """
    from aami import is_beat_symbol

    ann_samples = np.asarray(ann_samples, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    names = ["rr_prev", "rr_next", "rr_mean_local", "rr_ratio", "prematurity_ratio"]

    if target_indices.size == 0:
        return np.empty((0, 5)), names, np.zeros(0, dtype=bool)

    # Positions of real beats within the annotation series
    beat_pos = np.array(
        [i for i, s in enumerate(ann_symbols) if is_beat_symbol(s)], dtype=np.int64
    )
    if beat_pos.size < 3:
        return np.empty((0, 5)), names, np.zeros(target_indices.size, dtype=bool)

    # RR series: rr[j] is the interval (seconds) between beat_pos[j] and beat_pos[j+1]
    rr = np.diff(ann_samples[beat_pos]) / float(fs)

    # Map annotation index -> position within the beat series
    pos_of = {int(p): j for j, p in enumerate(beat_pos)}

    rows = []
    keep = []
    for t in target_indices.tolist():
        k = pos_of.get(int(t))
        if k is None or k == 0 or k == len(beat_pos) - 1:
            keep.append(False)
            continue

        rr_prev = rr[k - 1]
        rr_next = rr[k]

        lo = max(0, k - local_window)
        hi = min(len(rr), k + local_window + 1)
        local = np.concatenate([rr[lo:k], rr[k + 1:hi]])
        rr_mean_local = float(np.mean(local)) if local.size else rr_prev

        # Guard against degenerate/zero intervals (annotation collisions)
        rr_prev = max(float(rr_prev), 1e-9)
        rr_next = max(float(rr_next), 1e-9)
        rr_mean_local = max(rr_mean_local, 1e-9)

        rr_ratio = rr_next / rr_prev
        prematurity_ratio = rr_prev / rr_mean_local

        rows.append([rr_prev, rr_next, rr_mean_local, rr_ratio, prematurity_ratio])
        keep.append(True)

    keep_mask = np.asarray(keep, dtype=bool)
    if not rows:
        return np.empty((0, 5)), names, keep_mask
    return np.asarray(rows, dtype=float), names, keep_mask


def extract_qrs_morphology_features(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    window_s: float = QRS_WINDOW_S
) -> Tuple[np.ndarray, list]:
    """
    Extract QRS morphology features from signal segments around R-peaks.

    Parameters
    ----------
    signal : np.ndarray
        ECG signal
    r_peaks : np.ndarray
        Sample indices of detected R-peaks
    fs : float
        Sampling frequency
    window_s : float
        Analysis window half-width in seconds (total window = 2 * window_s)

    Returns
    -------
    features : np.ndarray
        Shape (n_beats - 2, n_features)
    names : list of str
        Feature names
    """
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 3:
        return np.empty((0, 10)), [
            "qrs_amp", "qrs_width", "qrs_energy",
            "beat_mean", "beat_std", "beat_var",
            "beat_skew", "beat_kurtosis",
            "r_amp_rel", "rs_amp_ratio"
        ]

    w = int(round(window_s * fs))
    X = []

    for i, r in enumerate(r_peaks):
        if i == 0 or i == len(r_peaks) - 1:
            continue

        lo = max(0, r - w)
        hi = min(len(signal), r + w)
        seg = signal[lo:hi]

        if len(seg) < w:  # Boundary check
            continue

        # Baseline estimation from wider window
        base_lo = max(0, r - w * 3)
        base_hi = min(len(signal), r + w * 3)
        baseline = float(np.median(signal[base_lo:base_hi]))

        # Centered segment
        seg_centered = seg - baseline

        # QRS amplitude (peak absolute deviation)
        amp = float(np.max(np.abs(seg_centered)))

        # QRS width: duration above 30% of peak
        above = np.abs(seg_centered) > 0.30 * amp
        width = float(above.sum()) / fs

        # QRS energy: sum of squared derivative
        d = np.diff(seg_centered)
        energy = float(np.sum(d * d)) / (w * w)

        # Beat window statistics
        beat_mean = float(np.mean(seg_centered))
        beat_std = float(np.std(seg_centered))
        beat_var = float(np.var(seg_centered))
        beat_skew = float(skew(seg_centered, nan_policy="omit"))
        beat_kurt = float(kurtosis(seg_centered, nan_policy="omit"))

        # R-peak amplitude relative to baseline
        r_amp_rel = float(seg_centered[min(w, len(seg_centered) - 1)])

        # R/S amplitude ratio (if S wave exists)
        r_peak_idx = int(np.argmax(seg_centered))
        s_candidates = seg_centered[r_peak_idx:]
        if len(s_candidates) > 1:
            s_amp = float(np.min(s_candidates))  # Most negative after R
            rs_amp_ratio = abs(s_amp) / max(r_amp_rel, 1e-9) if r_amp_rel != 0 else 0.0
        else:
            rs_amp_ratio = 0.0

        X.append([
            amp, width, energy,
            beat_mean, beat_std, beat_var,
            beat_skew, beat_kurt,
            r_amp_rel, rs_amp_ratio
        ])

    names = [
        "qrs_amp", "qrs_width", "qrs_energy",
        "beat_mean", "beat_std", "beat_var",
        "beat_skew", "beat_kurtosis",
        "r_amp_rel", "rs_amp_ratio"
    ]
    return np.asarray(X, dtype=float), names


def extract_features(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    include_rr: bool = True,
    include_morphology: bool = True,
    window_s: float = QRS_WINDOW_S
) -> Tuple[np.ndarray, list]:
    """
    Extract all beat-level features.

    Parameters
    ----------
    signal : np.ndarray
        ECG signal
    r_peaks : np.ndarray
        Sample indices of detected R-peaks
    fs : float
        Sampling frequency
    include_rr : bool
        Include RR-interval features
    include_morphology : bool
        Include QRS morphology features
    window_s : float
        Analysis window half-width in seconds

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_beats, n_features)
    feature_names : list of str
        Names of features in order
    """
    r_peaks = np.asarray(r_peaks, dtype=int)

    features_list = []
    names_list = []

    if include_rr:
        rr_feat, rr_names = extract_rr_features(r_peaks, fs)
        features_list.append(rr_feat)
        names_list.extend(rr_names)

    if include_morphology:
        morph_feat, morph_names = extract_qrs_morphology_features(
            signal, r_peaks, fs, window_s
        )
        features_list.append(morph_feat)
        names_list.extend(morph_names)

    if not features_list:
        return np.empty((0, 0)), []

    # All feature arrays should have same number of rows (beats with neighbors)
    X = np.hstack(features_list)
    return X, names_list


def extract_features_from_beats(
    beats: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, list]:
    """
    Extract features from pre-segmented beats (annotation-centered).

    This is used for the AAMI classification pipeline where beats are
    segmented around ground-truth annotation positions.

    Parameters
    ----------
    beats : np.ndarray
        Shape (n_beats, window_samples) - pre-segmented beats
    r_peaks : np.ndarray
        Original R-peak positions (for RR features)
    fs : float
        Sampling frequency
    labels : list of str, optional
        AAMI class labels (not used as features, but for alignment)

    Returns
    -------
    X : np.ndarray
        Feature matrix
    feature_names : list of str
    """
    # For pre-segmented beats, we can't easily compute RR intervals
    # since we don't have continuous peak positions.
    # Instead, compute morphology features directly from beat windows.

    if len(beats) == 0:
        return np.empty((0, 9)), [
            "qrs_amp", "qrs_width", "qrs_energy",
            "beat_mean", "beat_std", "beat_var",
            "beat_skew", "beat_kurtosis",
            "r_amp_rel"
        ]

    X = []
    for beat in beats:
        # Baseline is median of the beat window
        baseline = np.median(beat)
        seg_centered = beat - baseline

        # QRS amplitude
        amp = float(np.max(np.abs(seg_centered)))

        # QRS width (30% threshold)
        above = np.abs(seg_centered) > 0.30 * amp
        width = float(above.sum()) / fs

        # Energy
        d = np.diff(seg_centered)
        energy = float(np.sum(d * d)) / (len(beat) * len(beat))

        # Statistics
        beat_mean = float(np.mean(seg_centered))
        beat_std = float(np.std(seg_centered))
        beat_var = float(np.var(seg_centered))
        beat_skew = float(skew(seg_centered, nan_policy="omit"))
        beat_kurt = float(kurtosis(seg_centered, nan_policy="omit"))

        # R-peak relative amplitude (center of window)
        center_idx = len(beat) // 2
        r_amp_rel = float(seg_centered[center_idx])

        X.append([amp, width, energy, beat_mean, beat_std,
                  beat_var, beat_skew, beat_kurt, r_amp_rel])

    names = [
        "qrs_amp", "qrs_width", "qrs_energy",
        "beat_mean", "beat_std", "beat_var",
        "beat_skew", "beat_kurtosis",
        "r_amp_rel"
    ]
    return np.asarray(X, dtype=float), names


if __name__ == "__main__":
    # Quick test
    from generate_signal import generate_ecg
    from detect_rpeaks import detect_rpeaks

    signal, r_true, labels, fs = generate_ecg(duration_s=30, fs=250, seed=1)
    r_det = detect_rpeaks(signal, fs)

    X, names = extract_features(signal, r_det, fs)
    print(f"Features shape: {X.shape}")
    print(f"Feature names: {names}")
    print(f"First 3 rows:\n{X[:3]}")