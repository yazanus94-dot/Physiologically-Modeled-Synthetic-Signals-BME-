"""
ECG R-peak reference detection (WFDB XQRS) and ECG <-> PPG alignment.

Documentation of the workflow
-----------------------------
1. Detect R-peaks on a raw ECG lead with ``wfdb.processing.xqrs_detect``
   (the WFDB XQRS detector -- a wavelet-based detector, Pan-Tompkins
   lineage).  XQRS runs on the unfiltered lead using its own internal
   preprocessing; parameters are left at WFDB defaults and the sampling
   frequency is passed explicitly from the record header.

2. *** There are NO manual beat annotations in the BIDMC database. ***
   The XQRS output is an *automated detector's* output, not manual
   ground truth.  It is used here as an independent reference detector
   (different signal modality, different algorithm) against which the
   PPG detector is compared -- disagreement between the two does NOT
   by itself identify which detector is wrong.

3. *** Pulse transit time (PTT) ***: an ECG R-wave triggers ventricular
   depolarization; the resulting pressure wave reaches the pulse-oximeter
   measurement site after the ejection + pulse-wave travel time, i.e.
   the PPG systolic peak LAGS the R-wave by roughly 100-350 ms
   (subject-, site- and state-dependent).  R-peak timestamps and PPG
   peak timestamps must therefore NEVER be treated as simultaneous and
   never be matched 1:1 without allowing a lag.

   Strategy used here: compare *interval series* (RR intervals from ECG
   vs pulse-to-pulse intervals from PPG).  Both series describe the same
   underlying beat-to-beat timing, so they are compared after estimating
   a global lag between the two interval time series by
   cross-correlation on a common uniform time base.  This gives a
   lag-corrected correlation of the variability patterns -- it is NOT a
   claim of beat-by-beat simultaneity, and individual matched intervals
   are only used for *flagging* suspicious intervals (see
   quality_assessment.py), never as proof that one modality is correct.

Manual/visual sanity-check notes (recommended, not automated)
-------------------------------------------------------------
* Plot a 10-30 s ECG strip with the detected R-peaks overlaid and
  eyeball: (a) peaks land on QRS complexes, not T-waves or noise;
  (b) no grossly missed beats during arrhythmia/artifact;
  (c) the PPG strip below shows peaks occurring AFTER the R-waves by a
  plausible PTT (a few hundred ms), never before.
* XQRS confidence values can be inspected via ``return_conf=True``;
  low-confidence detections deserve visual review.
* None of this has been done automatically here -- no manual annotation
  is claimed for BIDMC.
"""
import numpy as np
from wfdb import processing as wfdb_processing


def detect_r_peaks(ecg, fs, verbose=False):
    """Detect ECG R-peaks with WFDB XQRS (``xqrs_detect``).

    Parameters
    ----------
    ecg : array-like
        Single-lead ECG in physical units (mV), as loaded from the
        record header.
    fs : float
        Sampling frequency in Hz (from the WFDB header, never hardcoded).

    Returns
    -------
    np.ndarray of int : R-peak sample indices.

    Raises
    ------
    ValueError
        If the signal is empty/all-invalid (caller should catch and
        record the failure instead of fabricating peaks).
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.size == 0 or not np.isfinite(ecg).any():
        raise ValueError("ECG signal empty or entirely invalid; cannot "
                         "detect R-peaks")
    # XQRS needs finite input: repair isolated NaN by interpolation
    if not np.isfinite(ecg).all():
        idx = np.arange(ecg.size)
        bad = ~np.isfinite(ecg)
        ecg = ecg.copy()
        ecg[bad] = np.interp(idx[bad], idx[~bad], ecg[~bad])
    peaks = wfdb_processing.xqrs_detect(sig=ecg, fs=float(fs),
                                        verbose=bool(verbose))
    return np.asarray(peaks, dtype=int)


def rr_intervals(peaks, fs):
    """RR / pulse intervals from a peak series.

    Returns
    -------
    (t_mid, intervals_s) : interval mid-point times (s) and interval
    durations (s).  Non-positive intervals (duplicate/decaying peaks)
    are dropped.
    """
    peaks = np.sort(np.asarray(peaks, dtype=int))
    if peaks.size < 2:
        return np.array([]), np.array([])
    iv = np.diff(peaks) / float(fs)
    t_mid = (peaks[:-1] + peaks[1:]) / 2.0 / float(fs)
    keep = np.isfinite(iv) & (iv > 0)
    return t_mid[keep], iv[keep]


def _common_grid(t_a, v_a, t_b, v_b, grid_hz=10.0, max_lag_s=3.0):
    """Interpolate two interval series onto one uniform time base."""
    if t_a.size < 4 or t_b.size < 4:
        return None
    t0 = max(t_a.min(), t_b.min())
    t1 = min(t_a.max(), t_b.max())
    if t1 - t0 <= 2 * max_lag_s:
        return None
    t = np.arange(t0, t1, 1.0 / grid_hz)
    if t.size < 8:
        return None
    return (t, np.interp(t, t_a, v_a), np.interp(t, t_b, v_b))


def estimate_interval_lag(t_ecg, v_ecg, t_ppg, v_ppg, grid_hz=10.0,
                          max_lag_s=3.0):
    """Estimate the lag between the ECG RR and PPG pulse-interval series.

    Cross-correlates the two interval series after mean removal and
    interpolation onto a common uniform grid.  Positive return value
    means the PPG interval series is DELAYED relative to the ECG series
    (as expected from pulse transit time and detector latencies).

    ``grid_hz=10`` gives 0.1 s lag resolution: pulse transit time plus
    detector latency is typically 0.2-0.6 s, so a coarser grid (the
    original 1 Hz) quantized the lag to whole seconds and could report a
    lag 0.5 s off the true optimum, destroying the lag-corrected
    correlation between the two series.

    Returns
    -------
    float : lag in seconds, or NaN if the series are too short /
    non-overlapping.
    """
    g = _common_grid(t_ecg, v_ecg, t_ppg, v_ppg, grid_hz, max_lag_s)
    if g is None:
        return float("nan")
    t, a, b = g
    a = a - a.mean()
    b = b - b.mean()
    if not (np.any(a) and np.any(b)):
        return float("nan")
    max_lag_n = int(round(max_lag_s * grid_hz))
    # normalized cross-correlation of the overlapping segments at each lag
    best_lag, best_c = 0, -np.inf
    for lag in range(-max_lag_n, max_lag_n + 1):
        if lag >= 0:
            x = a[:a.size - lag] if lag else a
            y = b[lag:]
        else:
            x = a[-lag:]
            y = b[:b.size + lag]
        if x.size < 8:
            continue
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        if nx == 0 or ny == 0:
            continue
        c = float(np.dot(x, y)) / (nx * ny)
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag / grid_hz


def align_interval_series(t_ecg, v_ecg, t_ppg, v_ppg, grid_hz=10.0,
                          max_lag_s=3.0, lag=None):
    """Lag-align the two interval series for comparison.

    Returns (t_grid, ecg_on_grid, ppg_shifted_on_grid, lag_s) where
    ``ppg_shifted`` is the PPG series moved BACK by ``lag_s`` so that
    the two variability patterns are compared at matched (lag-corrected)
    times.  Returns arrays + NaN lag when alignment is impossible.
    """
    if lag is None or not np.isfinite(lag):
        lag = estimate_interval_lag(t_ecg, v_ecg, t_ppg, v_ppg,
                                    grid_hz=grid_hz, max_lag_s=max_lag_s)
    g = _common_grid(t_ecg, v_ecg, t_ppg, v_ppg, grid_hz, max_lag_s)
    if g is None:
        return (np.array([]), np.array([]), np.array([]), float("nan"))
    t, a, b = g
    if np.isfinite(lag) and lag != 0:
        # ``b`` must be moved BACK by ``lag`` so that a PPG series delayed
        # by ``lag`` lands on the ECG series, i.e. evaluate b at ``t + lag``
        # using b's own coordinates ``t``:
        #     b_shifted(t) = b(t + lag)
        #
        # The previous form was ``np.interp(t, t + lag, b)``, which asks
        # for b at ``t - lag`` — the opposite direction.  Because ``b`` is
        # already delayed by ``lag``, that compounded to a shift of
        # -2*lag: a lag of 0.30 s moved the series by -0.60 s (verified
        # directly: r(b, v(t-0.6)) = 1.0000 while r(b, v(t+0.3)) =
        # 0.9495).  The lag-corrected HRV/PRV correlation was therefore
        # computed against a wrongly shifted series.
        #
        # Only times where ``t + lag`` stays inside b's coordinate range
        # are kept, so no value is silently clamped to an endpoint (the
        # old version clamped the first ``lag`` seconds to b[0], which is
        # why the first samples came back constant).
        coords = t.copy()
        query = t + float(lag)
        lo, hi = float(coords.min()), float(coords.max())
        keep = (query >= lo) & (query <= hi)
        if not np.any(keep):
            return (np.array([]), np.array([]), np.array([]), float(lag))
        t = t[keep]
        a = a[keep]
        b = np.interp(query[keep], coords, b)
    return t, a, b, float(lag)
