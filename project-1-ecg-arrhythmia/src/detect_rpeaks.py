"""
From-scratch Pan-Tompkins-style R-peak detector.

Pipeline (the classic 1985 loop, implemented with scipy instead of
analog hardware):
  bandpass (5-15 Hz) -> derivative -> squaring -> moving-window
  integration -> adaptive thresholding with a refractory period.

It is graded against the generator's ground-truth R peaks (sensitivity /
precision at a ±50 ms tolerance). A from-scratch detector is deliberately
not tuned to be artificially perfect — see the README for the honest
results discussion.
"""
import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt

REFRACTORY_S = 0.20  # physiological floor on consecutive R peaks


def _bandpass_legacy(signal, fs, low=2.0, high=15.0, order=3):
    """Legacy bandpass using (b, a) coefficients - kept for backward compatibility."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, padlen=min(len(signal) - 1, 3 * max(len(a), len(b))))


def _bandpass_sos(signal, fs, low=0.5, high=40.0, order=4):
    """Improved bandpass using second-order sections for numerical stability."""
    nyq = 0.5 * fs
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal, padlen=min(len(signal) - 1, 3 * order * 2))


def _derivative(x, fs):
    """Five-point derivative approximation."""
    k = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / (8.0 * (1.0 / fs))
    return np.convolve(x, k, mode="same")


def _moving_window_integral(x, fs, window_s=0.15):
    """Moving window integration."""
    w = max(1, int(round(window_s * fs)))
    return np.convolve(x, np.ones(w) / w, mode="same")


def detect_rpeaks(
    signal,
    fs,
    lowcut=2.0,
    highcut=15.0,
    filter_order=3,
    refractory_s=0.20,
    integration_window_s=0.15,
    qrs_threshold_factor=0.30,
    noise_threshold_factor=0.10,
    acceptance_factor=0.35,
    qrs_update_factor=0.4,
    noise_update_factor=0.1,
    use_sos=False
):
    """
    Detect R-peaks using Pan-Tompkins algorithm.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal
    fs : float
        Sampling frequency in Hz
    lowcut : float
        Low cutoff frequency for bandpass (Hz)
    highcut : float
        High cutoff frequency for bandpass (Hz)
    filter_order : int
        Filter order
    refractory_s : float
        Refractory period in seconds
    integration_window_s : float
        Moving window integration window in seconds
    qrs_threshold_factor : float
        Factor for QRS peak in threshold calculation
    noise_threshold_factor : float
        Factor for noise peak in threshold calculation
    acceptance_factor : float
        Minimum peak height as fraction of QRS peak for acceptance
    qrs_update_factor : float
        Learning rate for QRS peak estimate
    noise_update_factor : float
        Learning rate for noise peak estimate
    use_sos : bool
        If True, use second-order sections (more stable)

    Returns
    -------
    np.ndarray
        Sample indices of detected R peaks
    """
    # Bandpass filter
    if use_sos:
        b = _bandpass_sos(signal, fs, low=lowcut, high=highcut, order=filter_order)
    else:
        b = _bandpass_legacy(signal, fs, low=lowcut, high=highcut, order=filter_order)

    # Derivative
    d = _derivative(b, fs)

    # Squaring
    sq = d * d

    # Moving window integration
    mwi = _moving_window_integral(sq, fs, window_s=integration_window_s)

    refractory = int(refractory_s * fs)
    peaks, last_peak = [], -refractory

    # Adaptive thresholds
    qrs_pk = float(np.max(mwi[:fs]))
    noise_pk = float(np.median(mwi[:fs]) * 2.0)

    i = last_peak + refractory
    while i < len(mwi) - 1:
        window = mwi[i:i + fs]
        if len(window) == 0:
            break
        thresh = qrs_pk * qrs_threshold_factor + noise_pk * noise_threshold_factor
        above = np.where(window > thresh)[0]
        if len(above) == 0:
            i += fs
            continue
        c = int(above[0])
        lo, hi = c, min(c + refractory, len(window))
        cand = lo + int(np.argmax(window[lo:hi]))

        if i + cand - last_peak < refractory:
            i = i + cand + 1
            continue
        if window[cand] > acceptance_factor * qrs_pk:
            peaks.append(i + cand)
            last_peak = i + cand
            qrs_pk = (1 - qrs_update_factor) * qrs_pk + qrs_update_factor * float(window[cand])
            gap_lo = min(last_peak + 1, len(mwi) - 1)
            gap = mwi[gap_lo:gap_lo + refractory]
            if len(gap):
                noise_pk = (1 - noise_update_factor) * noise_pk + noise_update_factor * float(np.median(gap))
            i = last_peak + 1
        else:
            noise_pk = (1 - noise_update_factor) * noise_pk + noise_update_factor * float(window[cand])
            i = i + cand + 1

    # Deduplicate peaks within refractory window
    peaks = sorted(set(peaks))
    return np.array(peaks, dtype=int)


def detect_rpeaks_mitbih(signal, fs):
    """
    R-peak detection optimized for MIT-BIH (360 Hz) recordings.

    Uses wider bandpass (0.5-40 Hz) and SOS filtering for better
    handling of real-world noise and varying QRS morphologies.
    """
    return detect_rpeaks(
        signal, fs,
        lowcut=0.5,
        highcut=40.0,
        filter_order=4,
        refractory_s=0.20,
        integration_window_s=0.15,
        qrs_threshold_factor=0.30,
        noise_threshold_factor=0.10,
        acceptance_factor=0.35,
        qrs_update_factor=0.4,
        noise_update_factor=0.1,
        use_sos=True
    )


def match_peaks_one_to_one(detected, truth, fs, tolerance_s=0.05):
    """One-to-one matching of detected peaks against reference peaks.

    Each reference peak may be matched at most once and each detected peak
    at most once. Candidate pairs inside the tolerance window are considered
    in order of increasing temporal distance and accepted greedily, so a
    single detection can never account for two reference annotations.

    Returns
    -------
    list of (detected_idx, truth_idx, error_ms)
    """
    detected = np.asarray(detected, dtype=np.int64).ravel()
    truth = np.asarray(truth, dtype=np.int64).ravel()
    tol = int(round(tolerance_s * fs))

    if detected.size == 0 or truth.size == 0 or tol < 0:
        return []

    candidates = []
    for i, t in enumerate(truth):
        diffs = np.abs(detected - t)
        for j in np.nonzero(diffs <= tol)[0]:
            candidates.append((int(diffs[j]), int(j), int(i)))

    # Closest pair first; index tie-break keeps the result deterministic.
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    used_det = set()
    used_true = set()
    pairs = []
    for dist, j, i in candidates:
        if j in used_det or i in used_true:
            continue
        used_det.add(j)
        used_true.add(i)
        error_ms = (int(detected[j]) - int(truth[i])) * 1000.0 / float(fs)
        pairs.append((j, i, error_ms))

    pairs.sort(key=lambda p: p[1])  # return ordered by reference peak
    return pairs


def _score_from_pairs(pairs, n_det, n_true):
    """Derive tp/fp/fn, sensitivity, precision and f1 from one-to-one pairs."""
    tp = len(pairs)
    fp = int(n_det) - tp
    fn = int(n_true) - tp
    sens = tp / n_true if n_true else 0.0
    prec = tp / n_det if n_det else 0.0
    f1 = 2 * sens * prec / (sens + prec) if (sens + prec) > 0 else 0.0
    timing = [p[2] for p in pairs]
    return tp, fp, fn, sens, prec, f1, timing


def score_detector(detected, truth, fs, tolerance_s=0.05):
    """Sensitivity/precision of peak detection against ground truth.

    tolerance_s = matching window (default ±50 ms, as used by the classic
    R-peak detection literature). Matching is strictly one-to-one, so
    precision is always ≤ 1.0.
    """
    pairs = match_peaks_one_to_one(detected, truth, fs, tolerance_s)
    tp, fp, fn, sens, prec, _f1, _t = _score_from_pairs(
        pairs, np.asarray(detected).size, np.asarray(truth).size
    )
    return {"sensitivity": round(sens, 4), "precision": round(prec, 4),
            "tp": tp, "fp": fp, "fn": fn}


def score_detector_detailed(detected, truth, fs, tolerance_s=0.05):
    """
    Detailed detector scoring with timing error statistics.

    Returns
    -------
    dict with keys:
        sensitivity, precision, f1, tp, fp, fn
        mean_timing_error_ms, std_timing_error_ms
        matched_pairs: list of (detected_idx, truth_idx, error_ms)

    Notes
    -----
    Matching is strictly one-to-one (see ``match_peaks_one_to_one``): a
    detection already accounting for one reference annotation cannot be
    counted again, which keeps precision bounded by 1.0.
    """
    n_det = int(np.asarray(detected).size)
    n_true = int(np.asarray(truth).size)

    pairs = match_peaks_one_to_one(detected, truth, fs, tolerance_s)
    tp, fp, fn, sens, prec, f1, timing = _score_from_pairs(pairs, n_det, n_true)

    return {
        "sensitivity": round(sens, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_timing_error_ms": round(float(np.mean(timing)), 2) if timing else 0.0,
        "std_timing_error_ms": round(float(np.std(timing)), 2) if timing else 0.0,
        "matched_pairs": pairs
    }


if __name__ == "__main__":
    # Quick test with synthetic data
    from generate_signal import generate_ecg

    signal, r_true, labels, fs = generate_ecg(duration_s=30, fs=250, seed=1, noise=False)

    # Legacy detector
    r_det_legacy = detect_rpeaks(signal, fs, lowcut=2.0, highcut=15.0, use_sos=False)
    score_legacy = score_detector_detailed(r_det_legacy, r_true, fs)
    print("Legacy (2-15 Hz):", score_legacy)

    # Improved detector
    r_det_improved = detect_rpeaks(signal, fs, lowcut=0.5, highcut=40.0, use_sos=True)
    score_improved = score_detector_detailed(r_det_improved, r_true, fs)
    print("Improved (0.5-40 Hz SOS):", score_improved)

    # MIT-BIH optimized
    r_det_mitbih = detect_rpeaks_mitbih(signal, fs)
    score_mitbih = score_detector_detailed(r_det_mitbih, r_true, fs)
    print("MIT-BIH optimized:", score_mitbih)