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

    # full experiment (all 6 SNR levels, both records, 30 s segments)
    python run_real_experiment.py --download

    # re-run with data already present (no --download needed)
    python run_real_experiment.py

If the PhysioNet data has not been downloaded yet the runner prints a
clear message explaining how to get it and exits gracefully (exit code 0)
instead of failing.
"""
import argparse
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
from evaluate_denoising import (aggregate_metrics,
                                compute_segment_metrics, _zero_mean)
from load_real_ecg import (MITDB_DIR, download_records as download_ecg,
                           is_downloaded as ecg_is_downloaded,
                           list_remote_records, load_ecg_records,
                           local_records as local_ecg_records)
from load_real_noise import (NOISE_CATEGORIES, NOISE_CATEGORY_BY_RECORD,
                             NSTDB_DIR, discover_noise_records,
                             ensure_downloaded as ensure_noise_downloaded,
                             load_noise_records,
                             local_records as local_noise_records)
from visualization import (plot_metric_by_snr, plot_sample_comparison)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
DEFAULT_OUTDIR = os.path.join(PROJECT_DIR, "outputs", "real")
DEFAULT_ECG_RECORDS = ["100", "101"]

SUMMARY_METRICS = ["snr_db", "snr_improvement_db", "rmse", "nrmse", "prd",
                   "pearson", "rpeak_sensitivity", "rpeak_precision",
                   "rpeak_mae_ms", "qrs_amp_rel_change",
                   "qrs_width_rel_change"]


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
    p.add_argument("--duration", type=float, default=30.0,
                   help="seconds per evaluation segment (default 30)")
    p.add_argument("--segments", type=int, default=5,
                   help="independent noise segments per (noise x SNR x "
                        "record) cell (default 5; the audit requires >= 5)")
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
                                reference_peaks=ref_peaks)
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
                            "method": name,
                            "duration_s": float(n_ecg / fs_ecg),
                            "fs": float(fs_ecg),
                        }
                        row.update(metrics)
                        row["snr_improvement_db"] = (
                            metrics["snr_db"] - baseline)
                        rows.append(row)

                    if panel is None:
                        panel = (ecg_seg.copy(), noisy.copy(),
                                 {k: np.asarray(v).copy()
                                  for k, v in methods.items()},
                                 fs_ecg, achieved, noise["category"],
                                 target, ecg["name"])

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
    if df["rpeak_sensitivity"].notna().any():
        plot_metric_by_snr(
            rows, "rpeak_sensitivity",
            os.path.join(args.outdir, "rpeak_sensitivity_by_snr.png"),
            title="Real NSTDB noise: R-peak sensitivity after denoising",
            ylabel="Sensitivity (±50 ms, one-to-one)")
    if panel is not None:
        clean, noisy, methods, fs, achieved, cat, target, rec = panel
        plot_sample_comparison(
            clean, noisy, methods, fs,
            os.path.join(args.outdir, "sample_waveforms.png"),
            title=(f"Real ECG {rec} + {cat} noise mixed at {target} dB "
                   f"(first segment)"),
            achieved_snr_db=achieved)

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
    print(f"\nSaved -> {segments_csv}")
    print(f"Saved -> {summary_csv}")
    print(f"Saved figures -> {args.outdir}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
