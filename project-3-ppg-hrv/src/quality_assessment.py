"""
Signal-quality assessment and failure analysis for PPG-based PRV.

Investigates, per recording:

* motion disturbance        -- short-window energy excursions of the
                               pulse-band signal vs its own median level
* baseline variation        -- low-frequency drift metrics (from
                               preprocessing.assess_signal_quality)
* saturation / clipping     -- plateaus pinned at the rails
* low-amplitude pulses      -- beat-to-beat pulse amplitude collapse
* irregular pulse morphology-- correlation of each beat with the
                               record's median beat template
* missed beats              -- pulse intervals far longer than the local
                               norm (and ~2x the ECG RR when available)
* false peaks               -- pulse intervals far shorter than the norm
                               (and ~half the ECG RR when available)

When ECG is available, ECG-vs-PPG *interval* disagreement (lag-corrected
for pulse transit time, see ecg_reference.py) flags suspicious
intervals.  IMPORTANT: a disagreement does NOT automatically mean the
PPG detector failed -- the ECG detector (XQRS, also automated) can be
wrong too, and genuine ectopy makes the two modalities legitimately
disagree.  Every disagreement is therefore classified as
``missed_pulse_suspect`` / ``false_peak_suspect`` / ``ecg_suspect`` /
``uncertain`` and left for review; nothing is auto-attributed.

Emits a failure-analysis CSV (one row per suspicious interval or
per-record quality summary) -- see ``write_failure_csv``.
"""
import csv
import os

import numpy as np

from ppg_preprocessing import assess_signal_quality


def motion_stats(filtered_ppg, fs, window_s=2.0, k=3.0):
    """Detect motion disturbance as short-window energy excursions.

    Splits the (already bandpassed) PPG into ``window_s`` windows and
    flags windows whose std exceeds ``k`` times the median window std --
    motion artifact inflates local energy far above the pulse's own
    steady level.

    Returns dict: n_windows, motion_fraction, motion_windows (indices),
    median_window_std, k.
    """
    x = np.asarray(filtered_ppg, dtype=float)
    w = max(4, int(round(window_s * fs)))
    if x.size < 2 * w:
        return {"n_windows": 0, "motion_fraction": 0.0,
                "motion_windows": [], "median_window_std": float("nan"),
                "k": k, "window_s": window_s}
    n_win = x.size // w
    seg = x[:n_win * w].reshape(n_win, w)
    stds = np.nanstd(seg, axis=1)
    med = float(np.nanmedian(stds))
    if not np.isfinite(med) or med <= 0:
        return {"n_windows": n_win, "motion_fraction": 0.0,
                "motion_windows": [], "median_window_std": med,
                "k": k, "window_s": window_s}
    bad = np.flatnonzero(stds > k * med)
    return {"n_windows": n_win,
            "motion_fraction": float(bad.size / n_win),
            "motion_windows": bad.tolist(),
            "median_window_std": med, "k": k, "window_s": window_s}


def beat_morphology_stats(filtered_ppg, fs, peaks,
                          pre_s=0.30, post_s=0.50, min_corr=0.5,
                          min_beats=4):
    """Irregular pulse morphology: correlation with the median beat.

    Each detected peak defines a beat window [peak-pre_s, peak+post_s].
    A median template is built from all valid windows; each beat is
    correlated (Pearson) with it.  Beats with correlation < ``min_corr``
    are morphology outliers (motion-corrupted or mis-detected).

    Also reports pulse-amplitude spread (robust CV) for amplitude
    collapse detection.

    Returns dict: n_beats, n_valid, n_morphology_outliers,
    morphology_outlier_fraction, median_correlation,
    amplitude_median, amplitude_mad, amplitude_robust_cv, min_corr.
    """
    x = np.asarray(filtered_ppg, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    wins = []
    amps = []
    for p in peaks:
        a, b = p - pre, p + post
        if a < 0 or b > x.size:
            continue
        seg = x[a:b]
        if not np.isfinite(seg).all():
            continue
        wins.append(seg)
        amps.append(float(np.max(seg) - np.min(seg)))
    n_beats = int(peaks.size)
    n_valid = len(wins)
    if n_valid < min_beats:
        return {"n_beats": n_beats, "n_valid": n_valid,
                "n_morphology_outliers": 0, "morphology_outlier_fraction":
                float("nan"), "median_correlation": float("nan"),
                "amplitude_median": float("nan"),
                "amplitude_mad": float("nan"),
                "amplitude_robust_cv": float("nan"), "min_corr": min_corr}
    wins = np.asarray(wins)
    amps = np.asarray(amps)
    template = np.median(wins, axis=0)
    t_sd = template.std()
    corrs = []
    for seg in wins:
        s_sd = seg.std()
        if t_sd > 0 and s_sd > 0:
            c = float(np.corrcoef(seg, template)[0, 1])
        else:
            c = 1.0 if s_sd == 0 and t_sd == 0 else 0.0
        corrs.append(c)
    corrs = np.asarray(corrs)
    outliers = int(np.sum(corrs < min_corr))
    amp_med = float(np.median(amps))
    amp_mad = float(np.median(np.abs(amps - amp_med)))
    return {"n_beats": n_beats, "n_valid": n_valid,
            "n_morphology_outliers": outliers,
            "morphology_outlier_fraction": float(outliers / n_valid),
            "median_correlation": float(np.median(corrs)),
            "amplitude_median": amp_med, "amplitude_mad": amp_mad,
            "amplitude_robust_cv": float(amp_mad / amp_med) if amp_med else float("nan"),
            "min_corr": min_corr}


def interval_suspects(intervals_s, t_mid, min_hr_bpm=40, max_hr_bpm=200,
                      norm_ratio_short=0.6, norm_ratio_long=1.5):
    """Flag physiologically implausible / locally atypical pulse intervals.

    Hard limits: interval outside [60/max_hr, 60/min_hr] seconds is
    physiologically impossible for the configured HR search range.
    Soft limits: interval < 0.6 x median (false-peak suspect) or
    > 1.5 x median (missed-pulse suspect).

    Returns dict with suspect indices (into the interval arrays) and
    per-reason counts.
    """
    iv = np.asarray(intervals_s, dtype=float)
    t = np.asarray(t_mid, dtype=float)
    hard_lo, hard_hi = 60.0 / max_hr_bpm, 60.0 / min_hr_bpm
    med = float(np.nanmedian(iv)) if iv.size else float("nan")
    reasons = []
    idx = []
    for i, v in enumerate(iv):
        r = None
        if not np.isfinite(v) or v <= 0:
            r = "invalid_interval"
        elif v < hard_lo or v > hard_hi:
            r = "implausible_interval"
        elif np.isfinite(med) and med > 0 and v < norm_ratio_short * med:
            r = "false_peak_suspect"
        elif np.isfinite(med) and med > 0 and v > norm_ratio_long * med:
            r = "missed_pulse_suspect"
        if r:
            idx.append(i)
            reasons.append(r)
    counts = {k: int(reasons.count(k)) for k in
              ("invalid_interval", "implausible_interval",
               "false_peak_suspect", "missed_pulse_suspect")}
    return {"suspect_indices": idx, "reasons": reasons, "counts": counts,
            "n_intervals": int(iv.size), "median_interval_s": med,
            "hard_limits_s": [hard_lo, hard_hi]}


def compare_intervals(t_ppg, pi_s, t_ecg, rr_s, lag=None,
                      abs_tol_s=0.10, rel_tol=0.10,
                      min_hr_bpm=40, max_hr_bpm=200):
    """ECG-vs-PPG interval disagreement (lag-corrected for PTT).

    The ECG RR series is interpolated at each PPG interval mid-time,
    shifted by the cross-correlation lag (pulse transit time + detector
    latency, estimated in ecg_reference.estimate_interval_lag).

    Classification of a discrepancy (``|pi - rr| > max(abs_tol_s,
    rel_tol * rr)``) -- NOTE nothing is assumed to be the PPG's fault:

    * ``missed_pulse_suspect`` -- PPG interval ~2x the ECG RR (a pulse
      was likely missed by the PPG detector)
    * ``false_peak_suspect``    -- PPG interval ~half the ECG RR (an
      extra peak was likely inserted by the PPG detector)
    * ``ecg_suspect``           -- the ECG interval is itself outside
      physiological limits while the PPG interval is plausible (the
      automated ECG detector may be wrong)
    * ``uncertain``             -- genuine disagreement (ectopy,
      transient artifact, both detectors wrong); REQUIRES REVIEW --
      never auto-attributed to either modality

    Returns dict with counts, lag_s, and per-interval records.
    """
    from ecg_reference import estimate_interval_lag

    t_ppg = np.asarray(t_ppg, dtype=float)
    pi_s = np.asarray(pi_s, dtype=float)
    t_ecg = np.asarray(t_ecg, dtype=float)
    rr_s = np.asarray(rr_s, dtype=float)
    empty = {"lag_s": float("nan"), "n_compared": 0,
             "n_agree": 0, "n_disagree": 0,
             "counts": {"missed_pulse_suspect": 0, "false_peak_suspect": 0,
                        "ecg_suspect": 0, "uncertain": 0},
             "records": []}
    if t_ppg.size == 0 or t_ecg.size == 0 or pi_s.size < 2 or rr_s.size < 2:
        return empty
    if lag is None or not np.isfinite(lag):
        lag = estimate_interval_lag(t_ecg, rr_s, t_ppg, pi_s)
    hard_lo, hard_hi = 60.0 / max_hr_bpm, 60.0 / min_hr_bpm

    records = []
    counts = {"missed_pulse_suspect": 0, "false_peak_suspect": 0,
              "ecg_suspect": 0, "uncertain": 0}
    n_agree = 0
    for i, (tp, pv) in enumerate(zip(t_ppg, pi_s)):
        if not (np.isfinite(pv) and np.isfinite(tp)):
            continue
        # PPG-time tp corresponds to ECG-time tp - lag
        rr_at = float(np.interp(tp - lag, t_ecg, rr_s)) \
            if np.isfinite(lag) else float("nan")
        if not np.isfinite(rr_at) or rr_at <= 0:
            continue
        tol = max(abs_tol_s, rel_tol * rr_at)
        if abs(pv - rr_at) <= tol:
            n_agree += 1
            continue
        ratio = pv / rr_at
        if 1.6 <= ratio <= 2.5:
            cls = "missed_pulse_suspect"
        elif 0.4 <= ratio <= 0.63:
            cls = "false_peak_suspect"
        elif not (hard_lo <= rr_at <= hard_hi) and hard_lo <= pv <= hard_hi:
            cls = "ecg_suspect"
        else:
            cls = "uncertain"
        counts[cls] += 1
        records.append({"index": i, "t_ppg_s": float(tp),
                        "ppg_interval_s": float(pv),
                        "ecg_rr_s": rr_at, "diff_s": float(pv - rr_at),
                        "ratio": float(ratio), "class": cls})
    return {"lag_s": float(lag),
            "n_compared": n_agree + sum(counts.values()),
            "n_agree": n_agree,
            "n_disagree": sum(counts.values()),
            "counts": counts, "records": records}


def assess_record(record_name, raw_ppg, filtered_ppg, fs, peaks,
                  ecg_peaks=None, quality_raw=None, min_hr_bpm=40,
                  max_hr_bpm=200):
    """Full per-record quality assessment (returns a JSON-safe dict).

    Parameters
    ----------
    raw_ppg, filtered_ppg : arrays (raw sensor output and bandpassed)
    peaks : detected PPG systolic peaks (sample indices)
    ecg_peaks : detected ECG R-peaks or None
    quality_raw : optional precomputed output of
        preprocessing.assess_signal_quality on the raw signal.
    """
    from ecg_reference import rr_intervals

    if quality_raw is None:
        quality_raw = assess_signal_quality(raw_ppg, fs)

    motion = motion_stats(filtered_ppg, fs)
    morph = beat_morphology_stats(filtered_ppg, fs, peaks)

    t_pi, pi = rr_intervals(peaks, fs)
    iv_sus = interval_suspects(pi, t_pi, min_hr_bpm, max_hr_bpm)

    ecg_cmp = None
    if ecg_peaks is not None and len(ecg_peaks) >= 2:
        t_rr, rr = rr_intervals(ecg_peaks, fs)
        ecg_cmp = compare_intervals(t_pi, pi, t_rr, rr,
                                    min_hr_bpm=min_hr_bpm,
                                    max_hr_bpm=max_hr_bpm)

    # --- hard failure flags ------------------------------------------
    failures = []
    if not quality_raw.get("usable", True):
        failures.extend(quality_raw.get("flags", []))
    if np.isfinite(morph["morphology_outlier_fraction"]) and \
            morph["morphology_outlier_fraction"] > 0.30:
        failures.append("irregular_morphology")
    if motion["motion_fraction"] > 0.20:
        failures.append("motion")
    if iv_sus["counts"]["implausible_interval"] > 0.05 * max(
            1, iv_sus["n_intervals"]):
        failures.append("implausible_intervals")

    return {
        "record_name": record_name,
        "fs": fs,
        "n_peaks": int(len(peaks)),
        "raw_quality": {"flags": quality_raw.get("flags", []),
                        "usable": quality_raw.get("usable", False),
                        "metrics": quality_raw.get("metrics", {})},
        "motion": motion,
        "morphology": morph,
        "intervals": iv_sus,
        "ecg_ppg_disagreement": ecg_cmp,
        "failures": sorted(set(failures)),
        "passed_quality": len(set(failures)) == 0,
        "n_suspect_intervals": len(iv_sus["suspect_indices"]),
    }


def write_failure_csv(rows, path):
    """Write the failure-analysis CSV (one row per suspicious interval /
    quality event).  Always writes a header, even with no rows -- an
    empty file with no header would silently hide that nothing failed.

    ``rows``: iterable of dicts; union of keys forms the header.
    """
    rows = list(rows)
    fields = ["record_name", "t_ppg_s", "ppg_interval_s", "ecg_rr_s",
              "diff_s", "ratio", "class", "reason", "source",
              "note"]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path
