"""
Real-data ECG denoising experiment (MIT-BIH ECG + NSTDB noise).

For every noise type (baseline wander ``bw``, muscle artifact ``ma``,
electrode-motion artifact ``em``) × every target SNR
([-6, 0, 6, 12, 18, 24] dB by default) it:

  1. loads a clean ECG segment from the MIT-BIH Arrhythmia Database
     (``data/physionet/mitdb/``, downloaded on demand),
  2. loads a real noise segment from the corresponding NSTDB pure noise
     record (``data/physionet/nstdb/``),
  3. mixes them at the requested SNR (``noise_mixer``),
  4. runs all five denoising methods (median baseline removal,
     0.5–40 Hz Butterworth band-pass, powerline notch, FFT
     soft-threshold, combined pipeline),
  5. scores every method (RMSE / NRMSE / PRD / Pearson / R-peak
     sensitivity / precision / timing error / QRS morphology), and
  6. writes a segment-level CSV, an aggregated mean±std CSV, and
     figures to ``outputs/real/``.

Usage
-----
    # smoke test (small subset, few SNR levels)
    python run_real_experiment.py --download --limit 1 --snrs -6 6 12 \
        --duration 20 --segments 1

    # full experiment (all 6 SNR levels, both records, 10 s segments)
    python run_real_experiment.py --download

    # re-run with data already present (no --download needed)
    python run_real_experiment.py

If the PhysioNet data has not been downloaded yet the runner prints a
clear message explaining how to get it and exits gracefully (exit code 0)
instead of failing.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from denoise_methods import (bandpass_filter, combined_pipeline,
                             fft_soft_threshold_denoise,
                             median_filter_baseline_removal, notch_filter,
                             snr_db)
from noise_mixer import TARGET_SNR_LEVELS, mix_to_snr
from evaluate_denoising import (DEFAULT_PEAK_TOLERANCE_S, aggregate_metrics,
                                compute_segment_metrics, _zero_mean,
                                detect_r_peaks, rpeak_metrics)
from load_real_ecg import (MITDB_DIR, download_records as download_ecg,
                           is_downloaded as ecg_is_downloaded,
                           list_remote_records, load_ecg_records,
                           local_records as local_ecg_records)
from load_real_noise import (NOISE_CATEGORIES, NOISE_CATEGORY_BY_RECORD,
                             NSTDB_DIR, discover_noise_records,
                             ensure_downloaded as ensure_noise_downloaded,
                             load_noise_records,
                             local_records as local_noise_records)
from visualization import (plot_metric_by_snr, plot_sample_comparison,
                           plot_metric_summary, plot_morphology_summary,
                           plot_noise_type_comparison, plot_powerline_ablation,
                           plot_tolerance_sensitivity,
                           plot_report_waveforms, plot_report_rpeak)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
DEFAULT_OUTDIR = os.path.join(PROJECT_DIR, "outputs", "real")
DEFAULT_ECG_RECORDS = ["118", "119"]   # MIT-BIH records 118 & 119 (with .atr annotations)

SUMMARY_METRICS = ["snr_db", "snr_improvement_db", "rmse", "nrmse", "prd",
                   "pearson", "rpeak_sensitivity", "rpeak_precision",
                   "rpeak_f1", "rpeak_mae_ms",
                   "qrs_amp_rel_change", "qrs_width_rel_change",
                   "input_ac_snr_db", "output_ac_snr_db",
                   "snr_improvement_ac_db"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Real-data ECG denoising experiment (mitdb + nstdb)")
    p.add_argument("--records", default=None,
                   help="comma-separated mitdb ECG records "
                        f"(default: {','.join(DEFAULT_ECG_RECORDS)})")
    p.add_argument("--limit", type=int, default=None,
                   help="use at most N ECG records (smoke tests)")
    p.add_argument("--noise-types", default=None,
                   help="comma-separated noise records (bw,ma,em) or "
                        "categories (baseline_wander,muscle_artifact,"
                        "electrode_motion); default: all three")
    p.add_argument("--snrs", nargs="+", default=None,
                   help="target SNR levels in dB (default: "
                        f"{TARGET_SNR_LEVELS})")
    p.add_argument("--duration", type=float, default=10.0,
                   help="seconds per evaluation segment (default 10 — the "
                        "master configuration)")
    p.add_argument("--segments", type=int, default=10,
                   help="independent noise segments per (noise x SNR x "
                        "record) cell (default 10 — the master configuration)")
    p.add_argument("--channel", type=int, default=0,
                   help="ECG channel index (default 0)")
    p.add_argument("--noise-channel", type=int, default=0,
                   help="noise channel index (default 0)")
    p.add_argument("--ecg-offset", type=float, default=0.0,
                   help="start offset into each ECG record, seconds")
    p.add_argument("--seed", type=int, default=3,
                   help="seed for noise-window selection")
    p.add_argument("--powerline", type=float, default=50.0,
                   choices=[50.0, 60.0],
                   help="powerline notch frequency (default 50)")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR,
                   help="output directory for CSV + figures")
    p.add_argument("--extra-records", default=None,
                   help="comma-separated additional annotated MIT-BIH "
                        "records to run the OPTIONAL expanded evaluation "
                        "on (results written to outputs/real_expanded/, "
                        "separately identifiable from the primary "
                        "benchmark; primary records stay 118/119)")
    p.add_argument("--download", action="store_true",
                   help="download missing PhysioNet data first")
    return p.parse_args(argv)


def _parse_snrs(values):
    if values is None:
        return list(TARGET_SNR_LEVELS)
    out = []
    for v in values:
        for part in str(v).split(","):
            if part.strip():
                out.append(float(part))
    if not out:
        raise ValueError("empty --snrs list")
    return out


def _parse_noise_types(values):
    """Accept record names (bw, ma, em) or categories
    (baseline_wander, muscle_artifact, electrode_motion)."""
    if values is None:
        return list(NOISE_CATEGORIES)
    wanted = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if not part:
                continue
            if part in NOISE_CATEGORY_BY_RECORD:
                cat = NOISE_CATEGORY_BY_RECORD[part]
            elif part in NOISE_CATEGORIES:
                cat = part
            else:
                raise ValueError(
                    f"unknown noise type {part!r}; use one of "
                    f"{sorted(NOISE_CATEGORY_BY_RECORD)} or "
                    f"{list(NOISE_CATEGORIES)}")
            if cat not in wanted:
                wanted.append(cat)
    return wanted


def _records_present(db, names):
    have = set(local_records(db) if db != "nstdb"
               else local_noise_records())
    return [n for n in names if n in have]


def _files_ok(db, name):
    directory = NSTDB_DIR if db == "nstdb" else MITDB_DIR
    return all(os.path.isfile(os.path.join(directory, name + ext))
               for ext in (".hea", ".dat"))


def resolve_data(args):
    """Work out which records to use and make sure they exist locally.

    Returns ``(ecg_records, {category: noise_record})`` or ``None`` when
    the dataset is missing and cannot (or may not) be downloaded — the
    caller then skips gracefully.
    """
    # --- which noise records? ---
    try:
        available_by_cat = {}
        for name, cat in discover_noise_records().items():
            available_by_cat.setdefault(cat, name)
    except Exception as exc:  # noqa: BLE001 - offline, nothing local
        print(f"[SKIP] NSTDB noise records are unavailable: {exc}")
        print(f"       Looked in: {NSTDB_DIR}")
        print("       Download them with:  python run_real_experiment.py "
              "--download")
        return None

    wanted_cats = _parse_noise_types(args.noise_types)
    noise_by_cat = {}
    for cat in wanted_cats:
        if cat not in available_by_cat:
            print(f"[SKIP] no nstdb record provides noise category "
                  f"{cat!r} (discovered: {available_by_cat})")
            return None
        noise_by_cat[cat] = available_by_cat[cat]

    # --- which ECG records? ---
    if args.records:
        ecg_records = [r.strip() for r in str(args.records).split(",")
                       if r.strip()]
    else:
        ecg_records = list(DEFAULT_ECG_RECORDS)
    if args.limit is not None:
        ecg_records = ecg_records[:max(0, args.limit)]

    # validate ECG record names against PhysioNet's official list when
    # reachable; otherwise just try them locally
    try:
        remote = set(list_remote_records("mitdb"))
        unknown = [r for r in ecg_records if r not in remote]
        if unknown:
            print(f"[warn] not mitdb records: {unknown} — they will only "
                  f"work if already downloaded")
    except Exception as exc:  # noqa: BLE001
        print(f"  [info] could not verify mitdb record list ({exc}); "
              f"using local files only")

    missing_ecg = [r for r in ecg_records if not _files_ok("mitdb", r)]
    missing_noise = [r for r in noise_by_cat.values()
                     if not _files_ok("nstdb", r)]

    if missing_ecg or missing_noise:
        if not args.download:
            print("[SKIP] PhysioNet data is not fully downloaded.")
            if missing_ecg:
                print(f"       missing mitdb records: {missing_ecg}")
                print(f"       expected under: {MITDB_DIR}")
            if missing_noise:
                print(f"       missing nstdb records: {missing_noise}")
                print(f"       expected under: {NSTDB_DIR}")
            print("       Download with:  python run_real_experiment.py "
                  "--download")
            return None
        if missing_noise:
            print(f"Downloading nstdb records {missing_noise} ...")
            try:
                ensure_noise_downloaded(missing_noise)
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] nstdb download failed: {exc}")
                return None
        if missing_ecg:
            print(f"Downloading mitdb records {missing_ecg} "
                  f"(with .atr annotations) ...")
            try:
                download_ecg("mitdb", records=missing_ecg,
                             annotators=["atr"])
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] mitdb download failed: {exc}")
                return None
        still_ecg = [r for r in ecg_records if not _files_ok("mitdb", r)]
        still_noise = [r for r in noise_by_cat.values()
                       if not _files_ok("nstdb", r)]
        if still_ecg or still_noise:
            print(f"[FAIL] download incomplete: still missing "
                  f"mitdb={still_ecg} nstdb={still_noise}")
            return None

    return ecg_records, noise_by_cat


def run_tolerance_sensitivity(args, ecg_loaded, noise_by_cat):
    """Tolerance sensitivity analysis (P2.1).

    Recomputes R-peak detection for the bandpass method on real
    NSTDB-mixed segments and reports sensitivity / precision / F1 /
    mean-absolute-timing-error / FP / FN at **±50, ±100 and ±150 ms**
    (the predefined tolerance is also reported — the choice is documented,
    not made on the basis of which is highest).

    Results are written to ``outputs/real/tolerance_sensitivity.json`` and
    ``tolerance_sensitivity.png``.
    """
    noise_loaded = load_noise_records(
        [noise_by_cat[c] for c in sorted(noise_by_cat)],
        channel=args.noise_channel)
    if not noise_loaded:
        print("[SKIP] tolerance sensitivity analysis: no noise records.")
        return []

    rows = []
    seg_budget = 2                       # keep the analysis cheap
    for noise in noise_loaded:
        for target in TARGET_SNR_LEVELS:
            for ecg in ecg_loaded:
                fs_ecg = ecg["fs"]
                fs_noise = noise["fs"]
                n_ecg = int(round(args.duration * fs_ecg))
                n_noise = int(round(args.duration * fs_noise))
                for seg in range(min(seg_budget, args.segments)):
                    ecg_start = seg * n_ecg
                    ecg_seg = _slice(ecg["signal"], ecg_start, n_ecg)
                    if ecg_seg is None:
                        continue
                    rng = np.random.default_rng(
                        [args.seed, 9999 + int(target), seg,
                         len(noise["name"]), len(ecg["name"])])
                    noise_start = int(rng.integers(
                        0, max(1, len(noise["signal"]) - n_noise + 1)))
                    noise_seg = _slice(noise["signal"], noise_start, n_noise)
                    if noise_seg is None:
                        noise_seg = noise["signal"][:n_noise]
                    try:
                        noisy, _achieved = mix_to_snr(
                            ecg_seg, noise_seg, float(target),
                            fs_clean=fs_ecg, fs_noise=fs_noise)
                    except ValueError:
                        continue
                    _rp = ecg.get("r_peaks")
                    ref_peaks = np.asarray(_rp if _rp is not None else [], dtype=int)
                    ref_peaks = ref_peaks[(ref_peaks >= ecg_start) &
                                          (ref_peaks < ecg_start + n_ecg)]
                    ref_peaks = ref_peaks - ecg_start
                    est = bandpass_filter(noisy, fs_ecg)
                    for tol_ms in (50.0, 100.0, 150.0):
                        m = rpeak_metrics(
                            ref_peaks, detect_r_peaks(est, fs_ecg), fs_ecg,
                            tol_ms / 1000.0)
                        rows.append({
                            "noise_category": noise["category"],
                            "noise_record": noise["name"],
                            "ecg_record": ecg["name"],
                            "target_snr_db": float(target),
                            "tolerance_ms": float(tol_ms),
                            "method": "bandpass_0.5_40hz",
                            **m,
                        })

    if not rows:
        print("[SKIP] tolerance sensitivity: no rows generated.")
        return rows

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, "tolerance_sensitivity.json")
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    plot_tolerance_sensitivity(rows,
                               os.path.join(outdir,
                                            "tolerance_sensitivity.png"))
    print(f"Saved -> {out_json}")
    print("Saved figure -> tolerance_sensitivity.png")
    return rows


def run_expanded_eval(args, extra_records):
    """Expanded evaluation on additional annotated MIT-BIH records.

    Runs the same mixing + denoising + scoring loop as the primary
    experiment on ``extra_records`` (e.g. records 207, 214) and writes to
    ``outputs/real_expanded/`` — a **separate** output directory so the
    original benchmark is never mixed into or overwritten.  Rows carry a
    ``protocol_tag`` of ``"expanded_extra"`` and record IDs so the two
    are distinguishable.  The original records (118/119) are unaffected.
    """
    outdir = os.path.join(args.outdir, "..", "real_expanded") \
        if os.path.basename(args.outdir.rstrip("/")) == "real" \
        else args.outdir.replace("real", "real_expanded")
    outdir = os.path.abspath(outdir)
    ecg_loaded = load_ecg_records(extra_records, channel=args.channel)
    if not ecg_loaded:
        print(f"[SKIP] expanded eval: no readable ECG records {extra_records}")
        return 0
    noise_by_cat = {cat: name for name, cat in discover_noise_records().items()}
    noise_loaded = load_noise_records(
        [noise_by_cat[c] for c in sorted(noise_by_cat)],
        channel=args.noise_channel)
    if not noise_loaded:
        print("[SKIP] expanded eval: no noise records.")
        return 0

    os.makedirs(outdir, exist_ok=True)
    rows = []
    for noise in sorted(noise_loaded, key=lambda d: d["category"]):
        for target in TARGET_SNR_LEVELS:
            for ecg in ecg_loaded:
                fs_ecg = ecg["fs"]
                fs_noise = noise["fs"]
                n_ecg = int(round(args.duration * fs_ecg))
                n_noise = int(round(args.duration * fs_noise))
                for seg in range(args.segments):
                    ecg_start = seg * n_ecg
                    ecg_seg = _slice(ecg["signal"], ecg_start, n_ecg)
                    if ecg_seg is None:
                        continue
                    rng = np.random.default_rng(
                        [args.seed + 1, seg, int(target) + 100,
                         len(noise["name"]), len(ecg["name"])])
                    noise_start = int(rng.integers(
                        0, max(1, len(noise["signal"]) - n_noise + 1)))
                    noise_seg = _slice(noise["signal"], noise_start, n_noise)
                    if noise_seg is None:
                        noise_seg = noise["signal"][:n_noise]
                    try:
                        noisy, achieved = mix_to_snr(
                            ecg_seg, noise_seg, float(target),
                            fs_clean=fs_ecg, fs_noise=fs_noise)
                    except ValueError:
                        continue
                    _rp = ecg.get("r_peaks")
                    ref_peaks = np.asarray(_rp if _rp is not None else [], dtype=int)
                    ref_peaks = ref_peaks[(ref_peaks >= ecg_start) &
                                          (ref_peaks < ecg_start + n_ecg)]
                    ref_peaks = ref_peaks - ecg_start
                    for name, est in run_all_methods(noisy, fs_ecg,
                                                     args.powerline).items():
                        try:
                            metrics = compute_segment_metrics(
                                ecg_seg, est, fs_ecg,
                                reference_peaks=ref_peaks, mixture=noisy)
                        except Exception as exc:  # noqa: BLE001
                            continue
                        rows.append({
                            "noise_category": noise["category"],
                            "noise_record": noise["name"],
                            "ecg_record": ecg["name"],
                            "protocol_tag": "expanded_extra",
                            "segment": seg,
                            "target_snr_db": float(target),
                            "achieved_snr_db": float(achieved),
                            "method": name,
                            "duration_s": float(n_ecg / fs_ecg),
                            "fs": float(fs_ecg),
                            **metrics,
                        })

    if not rows:
        print("[SKIP] expanded eval: no rows generated.")
        return 0
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "results_segments.csv"), index=False)
    print(f"Saved -> {os.path.join(outdir, 'results_segments.csv')}")
    print(f"Expanded eval summary -> {outdir}")
    return 0


def generate_report(args, rows, panel, ecg_records, snr_levels, noise_names):
    """Write ``report.json`` + waveform/error/R-peak report figures
    (P2.4): clean vs noisy vs denoised + error waveform, and an R-peak
    overlay for the bandpass method on the panel segment."""
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    report_dir = os.path.join(outdir, "report")
    os.makedirs(report_dir, exist_ok=True)

    agg = aggregate_metrics(rows)
    rr = {
        "records": list(ecg_records),
        "nstdb_records": list(noise_names),
        "target_snr_db": list(snr_levels),
        "duration_s": float(args.duration),
        "segments_per_cell": int(args.segments),
        "seed": int(args.seed),
        "powerline_hz": float(args.powerline),
        "peak_tolerance_ms": DEFAULT_PEAK_TOLERANCE_S * 1000.0,
        "n_rows": len(rows),
        "aggregate": {k: v for k, v in agg.items()
                      if isinstance(v, dict) and "mean" in v},
        "figures": {
            "report_waveforms": os.path.join(
                "report", "report_waveforms.png"),
            "report_rpeak": os.path.join("report", "report_rpeak.png"),
        },
    }
    try:
        sys.path.insert(0, os.path.join(PROJECT_DIR, os.pardir))
        from versions import get_versions
        rr["versions"] = get_versions()
    except Exception:  # noqa: BLE001
        rr.pop("versions", None)

    if panel is not None:
        clean, noisy, methods, fs, achieved, cat, target, rec, ref_peaks = panel
        plot_report_waveforms(
            clean, noisy, methods, fs,
            os.path.join(report_dir, "report_waveforms.png"),
            achieved=achieved,
            title=(f"Report — real ECG {rec} + {cat} noise mixed at "
                   f"{target} dB (achieved {achieved:.2f} dB)"))
        est = np.asarray(methods.get("bandpass_0.5_40hz", noisy), dtype=float)
        det_peaks = detect_r_peaks(est, fs)
        plot_report_rpeak(
            clean, est, fs,
            os.path.join(report_dir, "report_rpeak.png"),
            method="bandpass_0.5_40hz",
            ref_peaks=np.asarray(ref_peaks, dtype=int),
            det_peaks=det_peaks,
            title=(f"Report — R-peak overlay on denoised output "
                   f"(record {rec}, {cat} @ {target} dB)"))
    with open(os.path.join(outdir, "report.json"), "w") as fh:
        json.dump(rr, fh, indent=2, default=float)
    print(f"Saved -> {os.path.join(outdir, 'report.json')}")
    print(f"Saved figure -> report/report_waveforms.png")
    print(f"Saved figure -> report/report_rpeak.png")


def run_all_methods(noisy, fs, powerline_hz):
    """The five denoising methods (+ the raw noisy baseline row)."""
    return {
        "raw_noisy": np.asarray(noisy, dtype=float),
        "median_filter": median_filter_baseline_removal(noisy, fs)[0],
        "bandpass_0.5_40hz": bandpass_filter(noisy, fs),
        f"notch_{int(powerline_hz)}hz": notch_filter(noisy, fs,
                                                     freq=powerline_hz),
        "fft_soft_threshold": fft_soft_threshold_denoise(noisy, fs),
        "combined_pipeline": combined_pipeline(noisy, fs,
                                               powerline_hz=powerline_hz),
    }


def run_powerline_ablation(args, ecg_loaded):
    """Controlled 50 Hz / 60 Hz mains-interference ablation (audit item 4).

    Generates controlled powerline interference on the clean ECG records
    (both standard mains frequencies, two amplitudes each) using the
    record's clean reference as the base, then evaluates the notch filter
    and the other denoise methods on these corrupted signals — separately
    from the three NSTDB noise experiments.  Results are written to
    ``powerline_ablation.json`` and ``powerline_ablation_*.png``.

    For each interference frequency the notch is evaluated at that
    frequency (the tuning that should help) and at the other mains
    frequency (the mismatch case that should not), so the ablation
    demonstrates *why* the notch needs to be tuned to the actual
    interference — but the pipeline's shared ``DEFAULT_PEAK_TOLERANCE_S``
    is used throughout and never tuned to the ablation data.
    """
    outdir = args.outdir
    rows = []
    for ecg in ecg_loaded:
        name = ecg["name"]
        fs = ecg["fs"]
        clean = np.asarray(ecg["signal"], dtype=float)
        n = min(int(round(args.duration * fs)), len(clean))
        clean_seg = _zero_mean(clean[:n])
        ref = np.asarray(ecg["r_peaks"] if ecg["r_peaks"] is not None
                         else [], dtype=int)
        ref = ref[(ref >= 0) & (ref < n)]
        t = np.arange(n) / fs

        for freq in (50.0, 60.0):
            for amp_mv in (0.05, 0.10):
                noisy = clean_seg + amp_mv * np.sin(2 * np.pi * freq * t)
                methods = {
                    f"notch_{int(freq)}hz":
                        notch_filter(noisy, fs, freq=freq),
                    f"notch_{int(60 if freq == 50 else 50)}hz":
                        notch_filter(noisy, fs, freq=60 if freq == 50 else 50),
                    "bandpass_0.5_40hz": bandpass_filter(noisy, fs),
                    "median_filter": median_filter_baseline_removal(noisy, fs)[0],
                }
                for meth, est in methods.items():
                    try:
                        m = compute_segment_metrics(
                            clean_seg, est, fs, reference_peaks=ref)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [skip] powerline ablation {meth} "
                              f"{freq} Hz on {name}: {exc}")
                        continue
                    # The ablation reference is the clean ECG itself, so the
                    # metric ``snr_db`` is the improvement over clean.
                    rows.append({"record": name, "freq_hz": freq,
                                 "amp_mv": amp_mv, "method": meth,
                                 "snr_improvement_db": m["snr_db"],
                                 **m})

    if not rows:
        print("[WARN] powerline ablation produced no rows.")
        return []

    df = pd.DataFrame(rows)
    ablation_json = os.path.join(outdir, "powerline_ablation.json")
    with open(ablation_json, "w") as fh:
        json.dump(df.to_dict(orient="records"), fh, indent=2, default=float)

    # Figures (audit item 5: separate powerline-ablation figure)
    plot_powerline_ablation(
        rows, os.path.join(outdir, "powerline_ablation_snr.png"),
        title="Powerline ablation (clean ECG + 50/60 Hz mains): "
              "mean SNR improvement per method")
    print(f"Saved -> {ablation_json}")
    print("Saved figure -> powerline_ablation_snr.png")
    return rows


def _slice(signal, start, n):
    if start < 0 or start + n > len(signal):
        return None
    return signal[start:start + n]


def run_experiment(args):
    resolved = resolve_data(args)
    if resolved is None:
        return 0
    ecg_records, noise_by_cat = resolved
    snr_levels = _parse_snrs(args.snrs)

    print(f"Loading ECG records {ecg_records} (channel {args.channel}) ...")
    ecg_loaded = load_ecg_records(ecg_records, channel=args.channel)
    if not ecg_loaded:
        print("[SKIP] no readable ECG records.")
        return 0
    noise_names = [noise_by_cat[c] for c in sorted(noise_by_cat)]
    print(f"Loading nstdb noise records {noise_names} ...")
    noise_loaded = load_noise_records(noise_names,
                                      channel=args.noise_channel)
    if not noise_loaded:
        print("[SKIP] no readable noise records.")
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    panel = None
    failures = 0
    achieved_errors = []

    for noise in sorted(noise_loaded, key=lambda d: d["category"]):
        for target in snr_levels:
            for ecg in ecg_loaded:
                fs_ecg = ecg["fs"]           # read from the WFDB header
                fs_noise = noise["fs"]
                n_ecg = int(round(args.duration * fs_ecg))
                n_noise = int(round(args.duration * fs_noise))
                for seg in range(args.segments):
                    ecg_start = (int(round(args.ecg_offset * fs_ecg))
                                 + seg * n_ecg)
                    ecg_seg = _slice(ecg["signal"], ecg_start, n_ecg)
                    if ecg_seg is None:
                        print(f"  [skip] ECG {ecg['name']} has no samples "
                              f"at segment {seg} offset {ecg_start}")
                        continue
                    rng = np.random.default_rng(
                        [args.seed, seg,
                         int(target) + 100,
                         len(noise["name"]), len(ecg["name"])])
                    noise_start = int(rng.integers(
                        0, max(1, len(noise["signal"]) - n_noise + 1)))
                    noise_seg = _slice(noise["signal"], noise_start, n_noise)
                    if noise_seg is None:
                        noise_seg = noise["signal"][:n_noise]

                    try:
                        noisy, achieved = mix_to_snr(
                            ecg_seg, noise_seg, target,
                            fs_clean=fs_ecg, fs_noise=fs_noise)
                    except ValueError as exc:
                        print(f"  [skip] mix failed "
                              f"({noise['category']} @ {target} dB, "
                              f"{ecg['name']}): {exc}")
                        failures += 1
                        continue
                    achieved_errors.append(abs(achieved - target))

                    ref_peaks = ecg["r_peaks"]
                    if ref_peaks is None:
                        ref_peaks = np.array([], dtype=int)
                    ref_peaks = np.asarray(ref_peaks, dtype=int)
                    ref_peaks = ref_peaks[(ref_peaks >= ecg_start) &
                                          (ref_peaks < ecg_start + n_ecg)]
                    ref_peaks = ref_peaks - ecg_start

                    try:
                        methods = run_all_methods(noisy, fs_ecg,
                                                  args.powerline)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [skip] denoising failed "
                              f"({noise['category']} @ {target} dB, "
                              f"{ecg['name']}): {exc}")
                        failures += 1
                        continue

                    # Same DC-removed convention as compute_segment_metrics,
                    # so improvement = out - in compares like with like.
                    baseline = snr_db(_zero_mean(ecg_seg), _zero_mean(noisy))
                    for name, est in methods.items():
                        try:
                            metrics = compute_segment_metrics(
                                ecg_seg, est, fs_ecg,
                                reference_peaks=ref_peaks, mixture=noisy)
                        except Exception as exc:  # noqa: BLE001
                            print(f"  [skip] metrics failed for {name}: "
                                  f"{exc}")
                            failures += 1
                            continue
                        row = {
                            "noise_category": noise["category"],
                            "noise_record": noise["name"],
                            "ecg_record": ecg["name"],
                            "segment": seg,
                            "target_snr_db": float(target),
                            "achieved_snr_db": float(achieved),
                            "achieved_power_ratio_snr_db": float(achieved),
                            "method": name,
                            "duration_s": float(n_ecg / fs_ecg),
                            "fs": float(fs_ecg),
                        }
                        row.update(metrics)
                        row["snr_improvement_db"] = (
                            metrics["snr_db"] - baseline)
                        row["snr_improvement_ac_db"] = metrics["snr_improvement_ac_db"] \
                            if np.isfinite(metrics["snr_improvement_ac_db"]) else float("nan")
                        rows.append(row)

                    if panel is None:
                        panel = (ecg_seg.copy(), noisy.copy(),
                                 {k: np.asarray(v).copy()
                                  for k, v in methods.items()},
                                 fs_ecg, achieved, noise["category"],
                                 target, ecg["name"], ref_peaks.copy())

            print(f"  done: {noise['category']} ({noise['name']}) @ "
                  f"{target} dB -> {len(rows)} rows so far")

    if not rows:
        print("[FAIL] no segments were processed — nothing to report.")
        return 1

    df = pd.DataFrame(rows)
    segments_csv = os.path.join(args.outdir, "results_segments.csv")
    df.to_csv(segments_csv, index=False)

    group_cols = ["noise_category", "noise_record", "target_snr_db",
                  "method"]
    summary_rows = []
    for keys, grp in df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_segments"] = len(grp)
        for col in SUMMARY_METRICS:
            if col in grp:
                row[f"{col}_mean"] = float(grp[col].mean())
                row[f"{col}_std"] = float(grp[col].std(ddof=0))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.outdir, "results_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # --- JSON: authoritative machine-readable record (audit item: F1 in
    #     JSON; records, methodology, tolerance, achieved-SNR verification)
    agg = aggregate_metrics(rows)
    metrics_json = {
        "records": list(ecg_records),
        "nstdb_records": noise_names,
        "duration_s": float(args.duration),
        "segments_per_cell": int(args.segments),
        "seed": int(args.seed),
        "powerline_hz": float(args.powerline),
        "peak_tolerance_ms": DEFAULT_PEAK_TOLERANCE_S * 1000.0,
        "tolerance_s": DEFAULT_PEAK_TOLERANCE_S,
        "n_rows": int(len(rows)),
        "summary": summary_rows,
        "aggregate": {k: v for k, v in agg.items()
                      if isinstance(v, dict) and "mean" in v},
        "achieved_snr_db": {
            "max_abs_dev": max(achieved_errors) if achieved_errors else None,
            "mean_abs_dev": float(np.mean(achieved_errors))
                            if achieved_errors else None,
        },
        "verified_achieved_vs_target": bool(
            achieved_errors and max(achieved_errors) < 1e-6),
    }
    metrics_csv = os.path.join(args.outdir, "metrics.json")
    with open(metrics_csv, "w") as fh:
        json.dump(metrics_json, fh, indent=2, default=float)

    # --- figures ---
    plot_metric_by_snr(
        rows, "snr_improvement_db",
        os.path.join(args.outdir, "snr_improvement_by_snr.png"),
        title="Real NSTDB noise: SNR improvement per method",
        ylabel="SNR improvement over raw noisy (dB)")
    plot_metric_by_snr(
        rows, "nrmse",
        os.path.join(args.outdir, "nrmse_by_snr.png"),
        title="Real NSTDB noise: NRMSE per method",
        ylabel="NRMSE (RMSE / reference range)")
    plot_metric_summary(rows, "rmse",
                        os.path.join(args.outdir, "rmse_by_method.png"),
                        title="Real NSTDB noise: RMSE per method",
                        ylabel="RMSE (mV)")
    plot_metric_summary(rows, "prd",
                        os.path.join(args.outdir, "prd_by_method.png"),
                        title="Real NSTDB noise: PRD per method",
                        ylabel="PRD (%)")
    plot_morphology_summary(rows,
                            os.path.join(args.outdir,
                                         "morphology_by_method.png"))
    plot_noise_type_comparison(
        rows, os.path.join(args.outdir, "noise_type_comparison.png"),
        title="Real NSTDB noise: mean SNR improvement per noise type")
    if df["rpeak_sensitivity"].notna().any():
        plot_metric_by_snr(
            rows, "rpeak_sensitivity",
            os.path.join(args.outdir, "rpeak_sensitivity_by_snr.png"),
            title="Real NSTDB noise: R-peak sensitivity after denoising",
            ylabel="Sensitivity (±150 ms, one-to-one)")
    if df["rpeak_f1"].notna().any():
        plot_metric_by_snr(
            rows, "rpeak_f1",
            os.path.join(args.outdir, "rpeak_f1_by_snr.png"),
            title="Real NSTDB noise: R-peak F1 (2·P·S/(P+S)) per method",
            ylabel="F1 = 2·precision·sensitivity/(precision + sensitivity)")
    if panel is not None:
        clean, noisy, methods, fs, achieved, cat, target, rec, ref_peaks = panel
        plot_sample_comparison(
            clean, noisy, methods, fs,
            os.path.join(args.outdir, "sample_waveforms.png"),
            title=(f"Real ECG {rec} + {cat} noise mixed at {target} dB "
                   f"(first segment)"),
            achieved_snr_db=achieved)
        # Reproducible experiment report (P2.4)
        generate_report(args, rows, panel, ecg_records, snr_levels,
                        noise_names)

    # --- powerline ablation (separate experiment, see
    #     run_powerline_ablation: audit item 4) ---
    run_powerline_ablation(args, ecg_loaded)

    # --- tolerance sensitivity analysis (P2.1) ---
    run_tolerance_sensitivity(args, ecg_loaded, noise_by_cat)

    # --- report ---
    n_methods = int(df["method"].nunique())
    n_segments = int(len(df) // n_methods) if n_methods else 0
    n_cells = int(df.groupby(
        ["noise_category", "target_snr_db", "ecg_record", "segment"]
    ).ngroups)
    print(f"\nSegments processed: {n_segments} "
          f"({n_methods} methods, {n_cells} cells)")
    if achieved_errors:
        print(f"Mixer accuracy: max |achieved - target| = "
              f"{max(achieved_errors):.3e} dB "
              f"(mean {float(np.mean(achieved_errors)):.3e} dB)")
    if failures:
        print(f"Skipped/failed cells: {failures}")

    pivot = df.pivot_table(
        index=["noise_category", "target_snr_db"],
        columns="method", values="snr_improvement_db", aggfunc="mean")
    print("\n=== Mean SNR improvement over raw noisy signal (dB) ===")
    print(pivot.round(2).to_string())

    pivot2 = df.pivot_table(
        index=["noise_category", "target_snr_db"],
        columns="method", values="nrmse", aggfunc="mean")
    print("\n=== Mean NRMSE (RMSE / reference range) ===")
    print(pivot2.round(4).to_string())

    agg = aggregate_metrics(rows)
    print(f"\nAggregated pearson r: mean={agg.get('pearson', {}).get('mean')}"
          f"  rpeak_sensitivity mean="
          f"{agg.get('rpeak_sensitivity', {}).get('mean')}")
    print(f"Aggregated rpeak_f1: mean={agg.get('rpeak_f1', {}).get('mean')}")
    print(f"Aggregated rpeak_precision: mean="
          f"{agg.get('rpeak_precision', {}).get('mean')}")
    print(f"\nPeak tolerance: {DEFAULT_PEAK_TOLERANCE_S*1000:.0f} ms "
          f"(predefined; not tuned to data)")
    print(f"\nSaved -> {segments_csv}")
    print(f"Saved -> {summary_csv}")
    print(f"Saved -> {metrics_csv}")
    print(f"Saved figures -> {args.outdir}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.extra_records:
        extra = [r.strip() for r in str(args.extra_records).split(",")
                 if r.strip()]
        return run_expanded_eval(args, extra)
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
