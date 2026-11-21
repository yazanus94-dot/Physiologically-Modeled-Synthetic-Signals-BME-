"""
Preprocessing and signal-quality assessment for real PPG (and ECG).

All functions take the sampling frequency as an argument -- nothing is
hardcoded -- so the same code works for the 100 Hz synthetic generator
and the 125 Hz BIDMC recordings.

Filter defaults and their physiological rationale
--------------------------------------------------
* Bandpass 0.5-8 Hz, 4th-order Butterworth, zero-phase (SOS +
  ``sosfiltfilt``):

  - PPG pulse rates of interest span 40-200 bpm = 0.67-3.3 Hz; the
    lower cutoff of 0.5 Hz sits just below that band and removes slow
    baseline wander (respiration, sensor motion, venous pooling)
    without attenuating the slowest plausible pulses.
  - The upper cutoff of 8 Hz preserves the systolic upstroke and the
    dicrotic-notch inflection (morphology content up to a few Hz)
    while suppressing high-frequency sensor/EMG noise.  Filtering
    above ~10 Hz would start to round the dicrotic notch; going
    below ~0.5 Hz would distort pulse shape and slow beats.
  - Zero-phase (forward-backward) filtering is used so pulse arrival
    times -- and therefore pulse intervals -- are NOT shifted, which
    matters for interval comparisons against ECG.

* Optional baseline correction: high-pass at 0.1 Hz (same zero-phase
  scheme) for records with strong residual drift after bandpass.

Normalization ('zscore' / 'minmax') is optional and off by default:
peak detection works on the bandpassed signal in native units, and
normalization must not be applied when absolute amplitude information
(e.g. low-perfusion checks) is needed.

Quality assessment flags (never applied silently -- they are reported):
clipping/saturation, flatline, low amplitude, excessive baseline
wander, invalid (NaN/inf) samples.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

# Default bandpass cutoffs (Hz) -- see module docstring for rationale.
DEFAULT_BAND_LO = 0.5
DEFAULT_BAND_HI = 8.0
DEFAULT_FILTER_ORDER = 4


def validate_signal(x, fs):
    """NaN/inf/length validation. Returns a dict of counts and fractions."""
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    n_nan = int(np.isnan(x).sum())
    n_inf = int(np.isinf(x).sum())
    n_finite = int(np.isfinite(x).sum())
    return {
        "n_samples": n,
        "duration_s": n / float(fs) if fs else float("nan"),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_finite": n_finite,
        "frac_invalid": (n_nan + n_inf) / n if n else 1.0,
        "all_invalid": n_finite == 0,
    }


def clean_invalid(x, fs, max_frac=0.5):
    """Replace NaN/inf by linear interpolation over finite samples.

    Returns (cleaned_x, info). If more than ``max_frac`` of samples are
    invalid (or everything is invalid) the signal cannot be repaired:
    ``info['repaired']`` is False and the cleaned array still contains
    NaN so callers must reject it.
    """
    x = np.asarray(x, dtype=float)
    info = validate_signal(x, fs)
    bad = ~np.isfinite(x)
    if not bad.any():
        return x.copy(), dict(info, repaired=True, n_interpolated=0)
    if info["frac_invalid"] > max_frac:
        return x.copy(), dict(info, repaired=False, n_interpolated=0)
    n = x.size
    idx = np.arange(n)
    x_out = x.copy()
    x_out[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x_out, dict(info, repaired=True, n_interpolated=int(bad.sum()))


def remove_dc(x, fs, method="median"):
    """Remove the DC offset of a signal (median or mean detrending).

    The median is robust to the pulse itself (pulses are brief), so it
    estimates the sensor offset without being biased by systolic peaks.
    """
    x = np.asarray(x, dtype=float)
    if method == "median":
        return x - np.nanmedian(x)
    if method == "mean":
        return x - np.nanmean(x)
    raise ValueError("unknown DC removal method: %r" % (method,))


def _design_bandpass(fs, low, high, order):
    nyq = 0.5 * fs
    low = float(low)
    high = float(high)
    if not (0 < low < high < nyq):
        # clip to a valid range rather than crashing on odd sampling rates
        high = min(high, 0.95 * nyq)
        low = min(low, 0.5 * high)
        if not (0 < low < high < nyq):
            raise ValueError("invalid bandpass for fs=%r: need "
                             "0 < low < high < fs/2" % fs)
    return butter(order, [low, high], btype="band", fs=fs, output="sos")


def bandpass(x, fs, low=DEFAULT_BAND_LO, high=DEFAULT_BAND_HI,
             order=DEFAULT_FILTER_ORDER, zero_phase=True):
    """Configurable Butterworth bandpass using second-order sections.

    ``zero_phase=True`` (default) applies ``sosfiltfilt`` so no phase
    delay is introduced (essential for interval timing); ``False``
    applies ``sosfilt`` (causal, has group delay -- avoid for timing).
    """
    x = np.asarray(x, dtype=float)
    sos = _design_bandpass(fs, low, high, order)
    if x.size < 16:
        raise ValueError("signal too short (%d samples) to filter" % x.size)
    if zero_phase:
        # padlen must leave at least one sample of signal
        default_pad = 3 * (2 * sos.shape[0] + 1)
        pad = min(default_pad, x.size - 1)
        return sosfiltfilt(sos, x, padlen=pad)
    from scipy.signal import sosfilt
    return sosfilt(sos, x)


def remove_baseline(x, fs, cutoff=0.1, order=2, zero_phase=True):
    """Optional baseline-wander correction: zero-phase high-pass at
    ``cutoff`` Hz (default 0.1 Hz -- below the 40-200 bpm pulse band)."""
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    cutoff = min(float(cutoff), 0.9 * nyq)
    sos = butter(order, cutoff, btype="high", fs=fs, output="sos")
    if zero_phase:
        pad = min(3 * (2 * sos.shape[0] + 1), x.size - 1)
        return sosfiltfilt(sos, x, padlen=pad)
    from scipy.signal import sosfilt
    return sosfilt(sos, x)


def normalize(x, method="zscore"):
    """Optional normalization: 'zscore', 'minmax' or 'none'."""
    x = np.asarray(x, dtype=float)
    if method == "none":
        return x.copy()
    if method == "zscore":
        sd = np.nanstd(x)
        if sd == 0 or not np.isfinite(sd):
            return x - np.nanmean(x)
        return (x - np.nanmean(x)) / sd
    if method == "minmax":
        lo, hi = np.nanmin(x), np.nanmax(x)
        if hi == lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)
    raise ValueError("unknown normalization method: %r" % (method,))


def _plateau_fraction(x, min_run_s, fs, side=None):
    """Fraction of samples belonging to constant runs >= ``min_run_s``.

    ``side`` restricts the plateau to the global max ('top') or min
    ('bottom') of the signal -- that is what saturation looks like in a
    clipped sensor output.  ``side=None`` measures any flat run.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    span = np.nanmax(x) - np.nanmin(x)
    if not np.isfinite(span) or span == 0:
        return 1.0
    if side == "top":
        level = np.nanmax(x)
    elif side == "bottom":
        level = np.nanmin(x)
    else:
        level = None
    if level is None:
        # any flat run: consecutive samples that do not change
        eq = np.concatenate(([False],
                             np.isclose(x[1:], x[:-1], rtol=0,
                                        atol=1e-12 * span),
                             [False]))
    else:
        tol = 1e-9 * abs(level)
        eq = np.abs(x - level) <= tol
    # eq marks samples that continue a plateau; find run boundaries
    runs = []
    start = None
    for i, v in enumerate(eq):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, n))
    min_run = max(2, int(round(min_run_s * fs)))
    in_run = np.zeros(n, dtype=bool)
    for a, b in runs:
        if b - a >= min_run:
            in_run[a:b] = True
    return float(in_run.mean())


def assess_signal_quality(x, fs, min_amplitude=None,
                          flatline_s=0.5, clip_run_s=0.05,
                          max_invalid_frac=0.05,
                          max_wander_ratio=1.0,
                          baseline_cutoff=0.35):
    """Assess raw-signal quality of a PPG (or ECG) waveform.

    Checks (all thresholds configurable, defaults documented here):

    * ``invalid``    -- NaN/inf sample fraction > ``max_invalid_frac``
                        (default 5%).
    * ``clipping``   -- saturation plateaus at the global max/min lasting
                        >= ``clip_run_s`` (default 50 ms) exceed 1% of the
                        record (sensor pinned at a rail).
    * ``flatline``   -- constant-value runs >= ``flatline_s`` (default
                        0.5 s) exceed 10% of the record (dead sensor /
                        disconnected lead).
    * ``low_amplitude`` -- robust amplitude (p99-p1) below
                        ``min_amplitude`` (default: only flagged when the
                        robust amplitude is ~0, i.e. effectively constant;
                        pass an absolute threshold in signal units for
                        stricter checks).  BIDMC PLETH is in normalized
                        units (observed p99-p1 ~ 0.2-0.6).
    * ``baseline_wander`` -- std of the < ``baseline_cutoff`` Hz baseline
                        component exceeds ``max_wander_ratio`` (default
                        1.0) times the std of the full signal (motion /
                        respiration drift dominating the pulse).

    Returns
    -------
    dict with per-check booleans, metrics, ``flags`` (list of failed
    check names) and ``usable`` (no hard failure).  This function only
    REPORTS -- it never modifies the signal.
    """
    x = np.asarray(x, dtype=float)
    v = validate_signal(x, fs)
    flags = []
    if v["frac_invalid"] > max_invalid_frac or v["all_invalid"]:
        flags.append("invalid")

    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {"flags": ["invalid", "flatline", "low_amplitude"],
                "usable": False, "validation": v,
                "metrics": {"robust_amplitude": float("nan")}}

    span = float(np.max(finite) - np.min(finite))
    robust_amp = float(np.percentile(finite, 99) - np.percentile(finite, 1))

    clip_frac = 0.0
    if span > 0:
        top = _plateau_fraction(finite, clip_run_s, fs, side="top")
        bot = _plateau_fraction(finite, clip_run_s, fs, side="bottom")
        clip_frac = max(top, bot)
    if clip_frac > 0.01:
        flags.append("clipping")

    flat_frac = _plateau_fraction(finite, flatline_s, fs, side=None)
    if flat_frac > 0.10:
        flags.append("flatline")

    if min_amplitude is None:
        low_amp = robust_amp <= 1e-12 * max(1.0, abs(float(np.median(finite))))
    else:
        low_amp = robust_amp < float(min_amplitude)
    if low_amp:
        flags.append("low_amplitude")

    # baseline wander: low-pass component vs total variability
    wander_ratio = float("nan")
    if span > 0 and finite.size >= int(4 * fs) and "invalid" not in flags:
        try:
            sos = butter(2, baseline_cutoff, btype="low", fs=fs, output="sos")
            pad = min(3 * (2 * sos.shape[0] + 1), x.size - 1)
            base = sosfiltfilt(sos, np.nan_to_num(x, nan=float(np.median(finite))),
                               padlen=pad)
            sd_sig = float(np.std(x[np.isfinite(x)]))
            sd_base = float(np.std(base))
            wander_ratio = sd_base / sd_sig if sd_sig > 0 else float("nan")
            if np.isfinite(wander_ratio) and wander_ratio > max_wander_ratio:
                flags.append("baseline_wander")
        except Exception:
            wander_ratio = float("nan")

    return {
        "validation": v,
        "metrics": {
            "robust_amplitude": robust_amp,
            "span": span,
            "clipping_fraction": float(clip_frac),
            "flatline_fraction": float(flat_frac),
            "baseline_wander_ratio": wander_ratio,
        },
        "flags": flags,
        "usable": len(flags) == 0,
    }


def preprocess_ppg(ppg, fs, low=DEFAULT_BAND_LO, high=DEFAULT_BAND_HI,
                   order=DEFAULT_FILTER_ORDER, dc_method="median",
                   baseline_correction=False, normalization="none"):
    """Full PPG preprocessing chain (returns filtered signal + info).

    1. validate + repair NaN/inf (interpolate; reject if >50% invalid)
    2. DC removal (median)
    3. Butterworth bandpass (SOS, zero-phase) -- morphology preserving
    4. optional baseline high-pass correction
    5. optional normalization (off by default)
    """
    x = np.asarray(ppg, dtype=float)
    cleaned, clean_info = clean_invalid(x, fs)
    if not clean_info["repaired"]:
        raise ValueError("signal has too many invalid samples "
                         "(%d NaN / %d inf of %d) to preprocess"
                         % (clean_info["n_nan"], clean_info["n_inf"],
                            clean_info["n_samples"]))
    x = remove_dc(cleaned, fs, method=dc_method)
    x = bandpass(x, fs, low=low, high=high, order=order, zero_phase=True)
    if baseline_correction:
        x = remove_baseline(x, fs)
    if normalization != "none":
        x = normalize(x, method=normalization)
    info = {"clean": clean_info, "band_hz": [low, high], "order": order,
            "zero_phase": True}
    return x, info
