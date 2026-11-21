"""
Real-data experiment: BIDMC PPG — HRV (ECG-derived) vs PRV (PPG-derived).

For every locally available BIDMC record:

  1. load PPG + ECG waveforms from the WFDB header (fs never hardcoded)
  2. preprocess the PPG (invalid-sample repair, DC removal, zero-phase
     bandpass) and detect systolic peaks
  3. detect ECG R-peaks with WFDB XQRS  ->  the **ECG-derived HRV**
     reference (RR intervals)
  4. pulse-to-pulse intervals from the PPG peaks  ->  **PPG-derived PRV**
  5. lag-align the two interval series (a constant inter-channel time
     offset is estimated by cross-correlation, never assumed)

Note on timing: in these BIDMC records the PLETH systolic peak falls at
the ECG R-peak (measured median offset -16..+16 ms) rather than the
~0.25 s later expected from pulse transit time.  That indicates a fixed
offset between the two recorded channels.  A *constant* offset cancels
exactly in interval-based analysis — RR and pulse intervals are measured
within each channel — so it shifts only the estimated ``lag_s`` and no
variability metric.  For that reason ``lag_s`` is reported as an
inter-channel alignment lag and is never described as a physiological
pulse transit time.
  6. compare time-domain (mean HR, SDNN, RMSSD, pNN50), frequency-domain
     (VLF/LF/HF, LF/HF) and beat-to-beat agreement, and — when the record
     carries 1 Hz monitor numerics — the monitor's own heart rate

Everything is reported under explicit ``ecg_hrv`` / ``ppg_prv`` keys so the
two quantities can never be conflated: HRV is computed from ECG R-R
intervals, PRV from PPG pulse intervals.  PRV ≈ HRV at rest is an empirical
finding of this run, not an assumption built into the code.

Run:  python run_bidmc_experiment.py [--download] [--records bidmc01 ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_bidmc import (DEFAULT_DATA_DIR, list_local_records,  # noqa: E402
                        load_record, download_records)
from ppg_preprocessing import preprocess_ppg, assess_signal_quality  # noqa: E402
from peak_detect import detect_peaks  # noqa: E402
from ecg_reference import (detect_r_peaks, rr_intervals,  # noqa: E402
                           estimate_interval_lag, align_interval_series)
from hrv_metrics import (time_domain_metrics, frequency_domain_metrics,  # noqa: E402
                         frequency_domain_metrics_from_intervals)
from quality_assessment import assess_record, compare_intervals  # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "outputs", "real")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="BIDMC real-data HRV (ECG) vs PRV (PPG) experiment")
    p.add_argument("--records", default=None, nargs="+",
                   help="record names (default: all present locally)")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N records")
    p.add_argument("--download", action="store_true",
                   help="fetch missing records from PhysioNet first")
    p.add_argument("--outdir", default=OUT_DIR,
                   help="output directory for metrics.json + figures")
    return p.parse_args(argv)


def _pct(a, b):
    """Relative difference in % of |b| (NaN-safe)."""
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)) or abs(b) < 1e-12:
        return float("nan")
    return 100.0 * (a - b) / abs(b)


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or a.size != b.size:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def _lag1(x):
    """Lag-1 autocorrelation of an interval series.

    Near zero or negative means successive intervals carry no shared
    structure — the series is dominated by independent beat-to-beat
    detection jitter rather than by smooth physiological variation.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return float("nan")
    x = x - x.mean()
    den = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / den) if den > 0 else float("nan")


def analyse_record(bundle):
    """Full HRV-vs-PRV analysis for one loaded BIDMC record."""
    name = bundle["record_name"]
    fs = float(bundle["fs"])
    out = {"record": name, "fs": fs,
           "duration_s": bundle["duration_s"],
           "warnings": list(bundle.get("warnings", []))}

    if bundle["ppg"] is None:
        raise ValueError("no PPG channel in record %s" % name)
    if bundle["ecg"] is None:
        raise ValueError("no ECG channel in record %s" % name)

    # ---------------- PPG: preprocess + systolic peaks -----------------
    raw_ppg = np.asarray(bundle["ppg"], dtype=float)
    raw_ecg = np.asarray(bundle["ecg"], dtype=float)

    quality_raw = assess_signal_quality(raw_ppg, fs)
    ppg_f, prep_info = preprocess_ppg(raw_ppg, fs)
    ppg_peaks = detect_peaks(ppg_f, fs)

    # ---------------- ECG: R-peaks (HRV reference) ---------------------
    ecg_peaks = detect_r_peaks(raw_ecg, fs)

    # ---------------- interval series ----------------------------------
    t_rr, rr_s = rr_intervals(ecg_peaks, fs)      # ECG-derived HRV input
    t_pi, pi_s = rr_intervals(ppg_peaks, fs)      # PPG-derived PRV input

    if rr_s.size < 16 or pi_s.size < 16:
        raise ValueError("too few intervals (rr=%d, pi=%d) in %s"
                         % (rr_s.size, pi_s.size, name))

    # Physiological-range artifact rejection (40-200 bpm), applied to BOTH
    # series before any variability metric: sensor dropouts and detector
    # dropouts otherwise enter as multi-second "intervals" and dominate
    # SDNN/RMSSD (a single 8 s gap inflates SDNN by hundreds of ms). This is
    # standard first-pass HRV cleaning; every rejection is counted and
    # reported, never silently discarded.
    iv_lo, iv_hi = 60.0 / 200.0, 60.0 / 40.0      # 0.30 s .. 1.50 s
    rr_ok = (rr_s >= iv_lo) & (rr_s <= iv_hi)
    pi_ok = (pi_s >= iv_lo) & (pi_s <= iv_hi)
    n_rr_rejected = int((~rr_ok).sum())
    n_pi_rejected = int((~pi_ok).sum())
    t_rr_f, rr_f = t_rr[rr_ok], rr_s[rr_ok]
    t_pi_f, pi_f = t_pi[pi_ok], pi_s[pi_ok]
    if rr_f.size < 16 or pi_f.size < 16:
        raise ValueError(
            "too few physiological intervals in %s after artifact "
            "rejection (rr %d/%d kept, pi %d/%d kept)"
            % (name, rr_f.size, rr_s.size, pi_f.size, pi_s.size))

    lag = estimate_interval_lag(t_rr_f, rr_f, t_pi_f, pi_f)
    t_grid, rr_grid, pi_grid, lag_s = align_interval_series(
        t_rr_f, rr_f, t_pi_f, pi_f, lag=lag)

    # ---------------- HRV (ECG) vs PRV (PPG) ---------------------------
    hrv_time = time_domain_metrics(rr_f * 1000.0)
    prv_time = time_domain_metrics(pi_f * 1000.0)
    hrv_freq = frequency_domain_metrics_from_intervals(t_rr_f, rr_f)
    prv_freq = frequency_domain_metrics_from_intervals(t_pi_f, pi_f)

    agreement = compare_intervals(t_pi_f, pi_f, t_rr_f, rr_f, lag=lag_s)
    n_cmp = max(1, agreement["n_compared"])
    agreement_pct = {
        "agree_pct": 100.0 * agreement["n_agree"] / n_cmp,
        "disagree_pct": 100.0 * agreement["n_disagree"] / n_cmp,
    }

    # lag-corrected beat-to-beat tracking between the two series
    track_r = _pearson(rr_grid, pi_grid) if t_grid.size else float("nan")

    # Smoothed (10 s window) tracking of the same two series.
    #
    # Both interval series measured here have a lag-1 autocorrelation near
    # zero or negative (-0.27 .. +0.12), i.e. each is dominated by
    # *independent* beat-to-beat peak-timing jitter rather than by shared
    # physiology.  Independent noise cancels in neither series, so the
    # raw beat-to-beat correlation is bounded by the ratio of shared
    # variance to total variance and stays low (measured -0.07 .. 0.39)
    # even when the two estimates of heart rate agree to 0.4 %.  Averaging
    # over 10 s suppresses that uncorrelated jitter and exposes the shared
    # heart-rate trend, which is the quantity HRV/PRV agreement is actually
    # about.  Both numbers are reported; neither is presented alone.
    def _smooth(t, v, w=10.0):
        v = np.asarray(v, dtype=float)
        out = np.empty_like(v)
        for i, ti in enumerate(t):
            m = (t >= ti - w / 2.0) & (t <= ti + w / 2.0)
            out[i] = v[m].mean() if m.any() else v[i]
        return out

    if t_grid.size >= 8:
        track_r_smoothed = _pearson(_smooth(t_grid, rr_grid),
                                    _smooth(t_grid, pi_grid))
    else:
        track_r_smoothed = float("nan")

    # ---------------- monitor numerics (independent reference) ---------
    monitor = None
    num = bundle.get("numerics")
    if num is not None:
        hr = num.get("hr")
        pulse = num.get("pulse")
        series = {}
        for key, vals in (("hr_bpm", hr), ("pulse_bpm", pulse)):
            if vals is None:
                continue
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v) & (v > 0)]
            if v.size:
                series[key] = round(float(v.mean()), 2)
        if series:
            monitor = {"mean": series,
                       "note": "bedside-monitor algorithm (1 Hz numerics), "
                               "not manual beat annotations"}

    out.update({
        "ppg_channel": bundle["ppg_channel"],
        "ecg_channel": bundle["ecg_channel"],
        "preprocessing": prep_info,
        "quality": quality_raw,
        "n_ppg_peaks": int(ppg_peaks.size),
        "n_ecg_rpeaks": int(ecg_peaks.size),
        "lag_s": lag_s,
        "interval_artifact_rejection": {
            "physiological_range_s": [iv_lo, iv_hi],
            "ecg_rr_kept": int(rr_f.size), "ecg_rr_rejected": n_rr_rejected,
            "ppg_pi_kept": int(pi_f.size), "ppg_pi_rejected": n_pi_rejected,
        },
        # ---- the two quantities, never conflated ----
        "ecg_hrv": {"source": "ECG R-R intervals (WFDB XQRS)",
                    "n_intervals_raw": int(rr_s.size),
                    "n_intervals_used": int(rr_f.size),
                    "time_domain": hrv_time,
                    "frequency_domain": hrv_freq},
        "ppg_prv": {"source": "PPG pulse-to-pulse intervals",
                    "n_intervals_raw": int(pi_s.size),
                    "n_intervals_used": int(pi_f.size),
                    "time_domain": prv_time,
                    "frequency_domain": prv_freq},
        # ---- HRV vs PRV comparison ----
        "hrv_prv_comparison": {
            "mean_hr_diff_pct": _pct(prv_time["mean_hr_bpm"],
                                     hrv_time["mean_hr_bpm"]),
            "sdnn_diff_pct": _pct(prv_time["sdnn_ms"], hrv_time["sdnn_ms"]),
            "rmssd_diff_pct": _pct(prv_time["rmssd_ms"],
                                   hrv_time["rmssd_ms"]),
            "pnn50_diff_pct": _pct(prv_time["pnn50_pct"],
                                   hrv_time["pnn50_pct"]),
            "lf_hf_ratio_diff_pct": _pct(prv_freq["lf_hf_ratio"],
                                         hrv_freq["lf_hf_ratio"]),
            "interval_series_pearson_lag_corrected": track_r,
            "interval_series_pearson_smoothed_10s": track_r_smoothed,
            # Lag-1 autocorrelation of the SUCCESSIVE-INTERVAL series
            # (rr_f / pi_f), not of the 10 Hz interpolated grid: on a grid
            # every interval is repeated across ~6 samples, so the value
            # measures interpolation smoothness (~0.95) and says nothing
            # about beat-to-beat structure.  Computed on the raw interval
            # series the numbers are near zero or negative, which is the
            # quantity the smoothing rationale above refers to.
            "lag1_autocorr_ecg_rr": _lag1(rr_f),
            "lag1_autocorr_ppg_pi": _lag1(pi_f),
            "agreement": agreement_pct,
            "disagreement_counts": agreement["counts"],
            "lag_s": lag_s,
        },
        "monitor_numerics": monitor,
        # keep raw series out of metrics.json (large); plots use them
        "_series": {"t_rr": t_rr_f, "rr_s": rr_f, "t_pi": t_pi_f,
                    "pi_s": pi_f, "t_grid": t_grid, "rr_grid": rr_grid,
                    "pi_grid": pi_grid},
    })
    out["quality_assessment"] = assess_record(
        name, raw_ppg, ppg_f, fs, ppg_peaks, ecg_peaks=ecg_peaks,
        quality_raw=quality_raw)
    return out


def _plot_record(res, out_dir):
    """Tachogram overlay + interval scatter for one record."""
    s = res["_series"]
    rec = res["record"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    axes[0].plot(s["t_rr"], s["rr_s"] * 1000.0, ".-", ms=3, lw=1,
                 color="#1B3A6B", label="ECG R-R (HRV reference)")
    axes[0].plot(s["t_pi"], s["pi_s"] * 1000.0, ".-", ms=3, lw=1,
                 color="#C2410C", label="PPG pulse interval (PRV)")
    axes[0].set_ylabel("interval (ms)")
    axes[0].set_title(f"{rec}: ECG-derived HRV vs PPG-derived PRV "
                      f"(lag = {res['lag_s']:.2f} s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    if s["t_grid"].size:
        axes[1].plot(s["t_grid"], s["rr_grid"] * 1000.0, "-",
                     color="#1B3A6B", lw=1.4, label="ECG (lag-aligned)")
        axes[1].plot(s["t_grid"], s["pi_grid"] * 1000.0, "--",
                     color="#C2410C", lw=1.4, label="PPG (shifted by lag)")
        r = res["hrv_prv_comparison"]["interval_series_pearson_lag_corrected"]
        axes[1].set_title(f"lag-corrected overlay (pearson r = {r:.3f})")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("interval (ms)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"tachogram_{rec}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_aggregate(per_record, out_dir):
    """Grouped bars: SDNN & RMSSD, HRV (ECG) vs PRV (PPG) per record."""
    names = [r["record"] for r in per_record]
    if not names:
        return None
    x = np.arange(len(names))
    w = 0.2
    sdnn_h = [r["ecg_hrv"]["time_domain"]["sdnn_ms"] for r in per_record]
    sdnn_p = [r["ppg_prv"]["time_domain"]["sdnn_ms"] for r in per_record]
    rmssd_h = [r["ecg_hrv"]["time_domain"]["rmssd_ms"] for r in per_record]
    rmssd_p = [r["ppg_prv"]["time_domain"]["rmssd_ms"] for r in per_record]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 1.5 * w, sdnn_h, w, label="SDNN — ECG (HRV)", color="#1B3A6B")
    ax.bar(x - 0.5 * w, sdnn_p, w, label="SDNN — PPG (PRV)", color="#7FA3D6")
    ax.bar(x + 0.5 * w, rmssd_h, w, label="RMSSD — ECG (HRV)", color="#C2410C")
    ax.bar(x + 1.5 * w, rmssd_p, w, label="RMSSD — PPG (PRV)", color="#F0A26B")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("ms")
    ax.set_title("Time-domain variability: ECG-derived HRV vs PPG-derived PRV")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "hrv_prv_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_example_strip(bundle, res, out_dir):
    """One screen of raw PPG + ECG with detected peaks marked."""
    fs = res["fs"]
    n = min(int(20 * fs), bundle["sig_len"])
    t = np.arange(n) / fs
    from ppg_preprocessing import preprocess_ppg as _pp  # local: strip uses same chain
    ppg_f, _ = _pp(np.asarray(bundle["ppg"], dtype=float)[:n], fs)
    ecg = np.asarray(bundle["ecg"], dtype=float)[:n]
    ppg_peaks = detect_peaks(ppg_f, fs)
    ecg_peaks = detect_r_peaks(ecg, fs)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(t, ppg_f, color="#C2410C", lw=0.9)
    sel = ppg_peaks[ppg_peaks < n]
    axes[0].plot(sel / fs, ppg_f[sel], "v", color="black", ms=6,
                 label=f"PPG peaks ({sel.size} in 20 s strip)")
    axes[0].set_ylabel("PPG (filtered)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{res['record']}: real BIDMC waveforms, 20 s")

    axes[1].plot(t, ecg, color="#1B3A6B", lw=0.9)
    sel = ecg_peaks[ecg_peaks < n]
    axes[1].plot(sel / fs, ecg[sel], "o", color="black", ms=5,
                 label=f"R-peaks ({sel.size} in 20 s strip)")
    axes[1].set_ylabel("ECG (mV)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "example_waveforms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    records = args.records or list_local_records(DEFAULT_DATA_DIR)
    if not records and args.download:
        print("No local records — downloading BIDMC from PhysioNet ...")
        download_records(data_dir=DEFAULT_DATA_DIR)
        records = list_local_records(DEFAULT_DATA_DIR)
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit(
            "No BIDMC records found in %s. Run with --download "
            "(or fetch them from https://physionet.org/content/bidmc/)."
            % DEFAULT_DATA_DIR)

    print("=" * 64)
    print("BIDMC REAL-DATA EXPERIMENT: ECG-derived HRV vs PPG-derived PRV")
    print("=" * 64)
    print(f"Records: {records}")

    per_record, failures = [], []
    for name in records:
        try:
            bundle = load_record(name, data_dir=DEFAULT_DATA_DIR)
            res = analyse_record(bundle)
        except Exception as exc:  # noqa: BLE001 — record-level skip, no fabrication
            print(f"  [skip] {name}: {exc}")
            failures.append({"record": name, "error": str(exc)})
            continue
        per_record.append(res)
        c = res["hrv_prv_comparison"]
        print(f"  {name}: HR(ecg)={res['ecg_hrv']['time_domain']['mean_hr_bpm']} "
              f"bpm HR(ppg)={res['ppg_prv']['time_domain']['mean_hr_bpm']} bpm "
              f"SDNN {res['ecg_hrv']['time_domain']['sdnn_ms']}/"
              f"{res['ppg_prv']['time_domain']['sdnn_ms']} ms "
              f"lag={res['lag_s']:.2f}s "
              f"agree={c['agreement']['agree_pct']:.1f}% "
              f"r={c['interval_series_pearson_lag_corrected']:.3f}")

    if not per_record:
        raise SystemExit("No record could be analysed — see errors above.")

    # ---------------- aggregate ----------------------------------------
    def _mean(key_path):
        vals = []
        for r in per_record:
            v = r
            for k in key_path:
                v = v[k]
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        return round(float(np.mean(vals)), 3) if vals else float("nan")

    aggregate = {
        "n_records": len(per_record),
        "mean_hr_diff_pct": _mean(["hrv_prv_comparison", "mean_hr_diff_pct"]),
        "mean_sdnn_diff_pct": _mean(["hrv_prv_comparison", "sdnn_diff_pct"]),
        "mean_rmssd_diff_pct": _mean(["hrv_prv_comparison", "rmssd_diff_pct"]),
        "mean_lf_hf_ratio_diff_pct": _mean(
            ["hrv_prv_comparison", "lf_hf_ratio_diff_pct"]),
        "mean_interval_series_pearson": _mean(
            ["hrv_prv_comparison", "interval_series_pearson_lag_corrected"]),
        "mean_interval_series_pearson_smoothed_10s": _mean(
            ["hrv_prv_comparison", "interval_series_pearson_smoothed_10s"]),
        "mean_agreement_pct": _mean(
            ["hrv_prv_comparison", "agreement", "agree_pct"]),
        "mean_lag_s": _mean(["hrv_prv_comparison", "lag_s"]),
        "mean_ecg_hrv_sdnn_ms": _mean(["ecg_hrv", "time_domain", "sdnn_ms"]),
        "mean_ppg_prv_sdnn_ms": _mean(["ppg_prv", "time_domain", "sdnn_ms"]),
        "mean_ecg_hrv_rmssd_ms": _mean(["ecg_hrv", "time_domain", "rmssd_ms"]),
        "mean_ppg_prv_rmssd_ms": _mean(["ppg_prv", "time_domain", "rmssd_ms"]),
    }

    # ---------------- plots --------------------------------------------
    for res in per_record:
        try:
            _plot_record(res, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  [plot skipped] {res['record']}: {exc}")
    _plot_aggregate(per_record, out_dir)
    try:
        bundle = load_record(per_record[0]["record"], data_dir=DEFAULT_DATA_DIR)
        _plot_example_strip(bundle, per_record[0], out_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"  [strip skipped] {exc}")

    # ---------------- save ---------------------------------------------
    for r in per_record:
        r.pop("_series", None)          # large arrays stay out of the JSON
    payload = {
        "experiment": "bidmc_hrv_prv",
        "dataset": "BIDMC PPG and Respiration Dataset (PhysioNet)",
        "data_dir": DEFAULT_DATA_DIR,
        "definitions": {
            "ecg_hrv": "heart-rate variability computed from ECG R-R "
                       "intervals (WFDB XQRS detector)",
            "ppg_prv": "pulse-rate variability computed from PPG "
                       "pulse-to-pulse intervals",
        },
        "records": per_record,
        "failures": failures,
        "aggregate": aggregate,
    }
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)

    print("\n" + "-" * 64)
    print("AGGREGATE (real BIDMC, mean over %d records)" % aggregate["n_records"])
    print(f"  mean HR difference (PRV vs HRV) : {aggregate['mean_hr_diff_pct']:.2f} %")
    print(f"  mean SDNN difference            : {aggregate['mean_sdnn_diff_pct']:.2f} %")
    print(f"  mean RMSSD difference           : {aggregate['mean_rmssd_diff_pct']:.2f} %")
    print(f"  mean LF/HF ratio difference     : {aggregate['mean_lf_hf_ratio_diff_pct']:.2f} %")
    print(f"  interval-series pearson (lag-corr): {aggregate['mean_interval_series_pearson']:.3f}")
    print(f"  ... after 10 s smoothing          : "
          f"{aggregate['mean_interval_series_pearson_smoothed_10s']:.3f}")
    print(f"  beat agreement within tolerance : {aggregate['mean_agreement_pct']:.1f} %")
    print(f"  mean estimated lag              : {aggregate['mean_lag_s']:.2f} s")
    print(f"\nSaved -> {path}")
    print(f"Figures -> {out_dir}")
    return payload


if __name__ == "__main__":
    main()
