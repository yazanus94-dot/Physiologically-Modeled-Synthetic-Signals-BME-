"""
Quantitative denoising metrics beyond the original (SNR, RMSE) pair.

Formulas
--------
* **RMSE**      ``sqrt( mean( (ref - est)^2 ) )``
* **NRMSE**     ``RMSE / (max(ref) - min(ref))``  (range-normalised, so
  it is comparable across records with different amplitudes)
* **PRD**       ``100 * ||ref - est|| / ||ref||``  (percentage
  root-mean-square difference, the classic ECG-compression/denoising
  score; 0 % is a perfect reconstruction)
* **Pearson r** standard linear correlation coefficient between ref/est
* **R-peak sensitivity / precision** after denoising, using a *strict
  one-to-one* peak matching with a ±50 ms tolerance: each detected peak
  may match at most one reference peak and vice versa (greedy matching
  over all candidate pairs sorted by |timing difference|, so a detected
  peak can never be counted twice against two nearby references)
* **Mean |R-peak timing error|** in ms over the matched pairs
* **Morphology preservation** — relative change of the median QRS
  peak-to-peak amplitude and of the median QRS width (width at half
  prominence of the band-passed, rectified signal) between the clean
  reference and the denoised estimate.

``aggregate_metrics`` reduces a list of per-segment metric dicts to
mean/std across segments (population std, ddof=0 → 0.0 for a single
segment).
"""
import numpy as np
from scipy.signal import butter, find_peaks, peak_widths, sosfiltfilt

from denoise_methods import rmse, snr_db  # noqa: F401  (re-exported)

DEFAULT_PEAK_TOLERANCE_S = 0.05   # ±50 ms, one R-peak matching tolerance
MIN_REFRACTORY_S = 0.24           # plausible max heart rate ~250 bpm


def nrmse(reference, estimate):
    """Range-normalised RMSE: RMSE / (max - min of the reference)."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    rng = reference.max() - reference.min()
    if rng <= 0:
        raise ValueError("reference has zero range — NRMSE is undefined")
    return float(rmse(reference, estimate) / rng)


def prd(reference, estimate):
    """Percentage root-mean-square difference, in percent."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    denom = float(np.linalg.norm(reference))
    if denom == 0:
        raise ValueError("reference is all zeros — PRD is undefined")
    return float(100.0 * np.linalg.norm(reference - estimate) / denom)


def pearson_correlation(reference, estimate):
    """Pearson correlation coefficient between two equal-length arrays."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    if reference.size < 2:
        raise ValueError("need at least 2 samples for correlation")
    ref_c = reference - reference.mean()
    est_c = estimate - estimate.mean()
    denom = float(np.linalg.norm(ref_c) * np.linalg.norm(est_c))
    if denom == 0:
        raise ValueError("correlation undefined for a constant signal")
    return float(np.dot(ref_c, est_c) / denom)


def _bandpass(x, fs, low=5.0, high=40.0, order=3):
    """Small helper band-pass used by the peak detector / QRS width
    measure. Falls back to the raw signal when the segment is too short
    or the band is invalid for the sampling rate."""
    nyq = 0.5 * fs
    if not (0 < low < high < nyq):
        return np.asarray(x, dtype=float)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    x = np.asarray(x, dtype=float)
    padlen = 3 * (2 * sos.shape[0] + 1)
    if x.size <= padlen:
        return x.copy()
    return sosfiltfilt(sos, x)


def detect_r_peaks(signal, fs):
    """Simple, deliberately-transparent R-peak detector used for the
    *benchmark* only: 5–40 Hz band-pass, then ``find_peaks`` with a
    240 ms refractory period and a prominence of 0.3·std of the
    band-passed signal. It is not a clinical detector — it exists so the
    same detector sees clean vs. noisy vs. denoised signals."""
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError(f"fs must be positive and finite, got {fs!r}")
    x = np.asarray(signal, dtype=float)
    if x.size < 8:
        return np.array([], dtype=int)
    filtered = _bandpass(x, fs)
    distance = max(1, int(round(MIN_REFRACTORY_S * fs)))
    prominence = 0.3 * float(np.std(filtered))
    peaks, _ = find_peaks(filtered, distance=distance,
                          prominence=prominence)
    return peaks


def match_peaks(reference_peaks, detected_peaks, tolerance_samples):
    """Strict one-to-one matching of detected peaks to reference peaks.

    All candidate pairs with |ref - det| <= tolerance are sorted by the
    absolute timing difference (ties broken deterministically) and
    accepted greedily, skipping any pair whose reference *or* detection
    is already used. Every reference and every detection is therefore
    matched at most once.

    Returns a list of ``(reference_index, detection_index)`` pairs.
    """
    ref = np.asarray(reference_peaks, dtype=int).ravel()
    det = np.asarray(detected_peaks, dtype=int).ravel()
    tol = int(tolerance_samples)
    if tol < 0:
        raise ValueError(f"tolerance must be >= 0 samples, got {tol}")

    candidates = []
    for i, r in enumerate(ref):
        for j, d in enumerate(det):
            diff = abs(int(r) - int(d))
            if diff <= tol:
                candidates.append((diff, i, j))
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    used_ref, used_det, matches = set(), set(), []
    for diff, i, j in candidates:
        if i in used_ref or j in used_det:
            continue
        used_ref.add(i)
        used_det.add(j)
        matches.append((i, j))
    matches.sort()
    return matches


def rpeak_metrics(reference_peaks, detected_peaks, fs,
                  tolerance_s=DEFAULT_PEAK_TOLERANCE_S):
    """Sensitivity, precision and mean absolute timing error of the
    detected R-peaks against the reference peaks."""
    ref = np.asarray(reference_peaks, dtype=int).ravel()
    det = np.asarray(detected_peaks, dtype=int).ravel()
    tol_samples = int(round(tolerance_s * fs))
    matches = match_peaks(ref, det, tol_samples)
    tp = len(matches)
    n_ref, n_det = ref.size, det.size
    sensitivity = float(tp / n_ref) if n_ref else float("nan")
    precision = float(tp / n_det) if n_det else float("nan")
    if tp:
        errors = np.array([abs(int(ref[i]) - int(det[j]))
                           for i, j in matches], dtype=float)
        mae_ms = float(errors.mean() / fs * 1000.0)
    else:
        mae_ms = float("nan")
    return {
        "rpeak_sensitivity": sensitivity,
        "rpeak_precision": precision,
        "rpeak_mae_ms": mae_ms,
        "rpeak_tp": tp,
        "rpeak_fp": int(n_det - tp),
        "rpeak_fn": int(n_ref - tp),
        "n_ref_peaks": int(n_ref),
        "n_detected_peaks": int(n_det),
    }


def qrs_morphology_metrics(reference, estimate, reference_peaks, fs,
                           window_s=0.12):
    """Relative change in QRS morphology between reference and estimate.

    Amplitude: median peak-to-peak amplitude in a ±``window_s`` window
    around each reference R-peak. Width: median width at half prominence
    of the band-passed, rectified signal at the reference R-peaks.
    Both relative changes are ``(estimate - reference) / reference``
    (0 = perfect morphology preservation).
    """
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    peaks = np.asarray(reference_peaks, dtype=int).ravel()
    n = reference.size
    valid = peaks[(peaks >= 0) & (peaks < n)]
    if valid.size == 0 or estimate.size != n:
        return {"qrs_amp_rel_change": float("nan"),
                "qrs_width_rel_change": float("nan")}

    half = int(round(window_s * fs))

    def _median_amplitude(x):
        amps = []
        for p in valid:
            lo, hi = max(0, p - half), min(n, p + half + 1)
            if hi <= lo:
                continue
            seg = x[lo:hi]
            amps.append(seg.max() - seg.min())
        return float(np.median(amps)) if amps else float("nan")

    amp_ref = _median_amplitude(reference)
    amp_est = _median_amplitude(estimate)
    amp_rel = (amp_est - amp_ref) / amp_ref if amp_ref > 0 else float("nan")

    def _median_width_ms(x):
        bp = np.abs(_bandpass(x, fs))
        usable = peaks[(peaks > 0) & (peaks < bp.size - 1)]
        if usable.size == 0:
            return float("nan")
        widths, _, _, _ = peak_widths(bp, usable, rel_height=0.5)
        return float(np.median(widths) / fs * 1000.0)

    width_ref = _median_width_ms(reference)
    width_est = _median_width_ms(estimate)
    width_rel = (width_est - width_ref) / width_ref \
        if np.isfinite(width_ref) and width_ref > 0 else float("nan")

    return {
        "qrs_amp_rel_change": float(amp_rel),
        "qrs_width_rel_change": float(width_rel),
        "qrs_amp_ref": amp_ref,
        "qrs_amp_est": amp_est,
        "qrs_width_ms_ref": width_ref,
        "qrs_width_ms_est": width_est,
    }


def _zero_mean(x):
    """Remove the DC level of a signal.

    ECG recordings carry an arbitrary electrode/baseline offset (the clean
    MIT-BIH segment used by the real experiment has mean ~ -0.33 mV).
    Denoising chains that high-pass or median-filter legitimately remove
    that offset, so scoring the raw difference against a DC-inclusive
    reference would report a large constant error that carries **no
    morphology information** — it made a perfect filter look 23 dB worse
    than it was.  Waveform-error metrics are therefore computed on
    mean-removed signals; range, Pearson correlation and all peak/morphology
    metrics are unaffected by a constant shift.
    """
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def compute_segment_metrics(clean, estimate, fs, reference_peaks=None):
    """All metrics for one (clean, denoised) pair.

    ``rmse`` / ``nrmse`` / ``prd`` / ``snr_db`` are computed after removing
    the DC level from both signals (see ``_zero_mean``); ``pearson`` and the
    R-peak / QRS morphology metrics are already offset-invariant.
    """
    clean = np.asarray(clean, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if clean.shape != estimate.shape:
        raise ValueError("clean and estimate must have the same shape")
    clean_z = _zero_mean(clean)
    est_z = _zero_mean(estimate)
    metrics = {
        "rmse": rmse(clean_z, est_z),
        "nrmse": nrmse(clean_z, est_z),
        "prd": prd(clean_z, est_z),
        "pearson": pearson_correlation(clean, estimate),
        "snr_db": snr_db(clean_z, est_z),
    }
    if reference_peaks is not None:
        detected = detect_r_peaks(estimate, fs)
        metrics.update(rpeak_metrics(reference_peaks, detected, fs))
        metrics.update(qrs_morphology_metrics(clean, estimate,
                                              reference_peaks, fs))
    return metrics


def aggregate_metrics(rows):
    """Mean/std of every numeric column across per-segment metric dicts
    (population std, ddof=0 → 0.0 when a single segment was run)."""
    if not rows:
        raise ValueError("no metric rows to aggregate")
    summary = {}
    for key in rows[0]:
        values = [row[key] for row in rows
                  if key in row and isinstance(row[key], (int, float,
                                                          np.floating,
                                                          np.integer))]
        if len(values) != len(rows) or not values:
            continue
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            mean = std = float("nan")
        else:
            mean = float(finite.mean())
            std = float(finite.std(ddof=0)) if finite.size > 1 else 0.0
        summary[key] = {"mean": mean, "std": std}
    return summary
