"""Automated evaluator for the MIT-BIH + NSTDB ECG denoising benchmark.

The evaluator generates the test mixtures deterministically (seeded), runs
the submitted solution on each, checks output structure and numerical
validity, computes signal-processing metrics, and returns a weighted
overall score (see ``SCORING_RUBRIC`` in ``TASK.md``).

Data sources, in order of preference:

1. **Real PhysioNet data** — loaded from the Project 2 loader modules
   (``project-2-ecg-denoising/src/load_real_ecg.py`` and
   ``load_real_noise.py``) when they are importable and the data is
   present locally (``data/physionet``).
2. **Synthetic fallback** — if the PhysioNet data is unavailable (e.g. a
   fresh checkout running the tests without downloading), a deterministic
   synthetic ECG + noise mixture is generated instead so the evaluator is
   still runnable offline.  The structure, validity, signal-processing
   and edge-case checks work on both; the *performance* numbers differ.

The evaluator never uses the clean reference to *generate* model
features, and it never tunes a candidate's hyperparameters on the final
test set — evaluation is external to the solution.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
P2_SRC = REPO_ROOT / "project-2-ecg-denoising" / "src"

#: Scoring rubric (weights in [0, 1], summing to 1.0).
SCORING_RUBRIC = {
    "structure": 0.15,            # finite, 1-D, same shape, non-degenerate
    "numerical_validity": 0.10,   # no NaN/Inf anywhere
    "signal_processing": 0.25,    # 50 Hz suppression / band content / morphology
    "performance": 0.25,          # mean SNR improvement across cells
    "reproducibility": 0.15,      # deterministic across two runs
    "edge_robustness": 0.10,      # short segment, zero-power input
}

TARGET_SNR_LEVELS = [-6.0, 0.0, 6.0, 12.0, 18.0, 24.0]
NSTDB_NOISE_RECORDS = ["bw", "ma", "em"]
DEFAULT_RECORDS = ["118", "119"]
DEFAULT_SEGMENTS = 10
DEFAULT_SEED = 3
DEFAULT_TOLERANCE_MS = 150.0


def _import_optional(name):
    """Import a module from the repo src path if it exists and loads."""
    if not P2_SRC.is_dir():
        return None
    spec = importlib.util.spec_from_file_location(
        name, P2_SRC / f"{name}.py",
        submodule_search_locations=[])
    if spec is None or spec.loader is None:
        return None
    try:
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(name, mod)
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        return None
    return mod


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_real_data(records, snrs, segments, seed=DEFAULT_SEED,
                    override_dir=None):
    """Load real ECG + noise. Returns (cells, loader_info) or None.

    ``override_dir`` is passed to the repo loaders as ``data_dir`` (None
    uses the loaders' default).
    """
    try:
        if str(P2_SRC) not in sys.path:
            sys.path.insert(0, str(P2_SRC))
        import wfdb  # noqa: F401
        from load_real_ecg import load_ecg_records
        from load_real_noise import load_noise_records
        from noise_mixer import mix_to_snr
    except Exception:  # noqa: BLE001
        return None, {}

    data_dir = override_dir
    if data_dir:
        # Implement `data_dir` by pointing the repo loaders' module-level
        # directory constants at the requested root.  The loaders derive
        # paths from these constants at call time (`records_dir` /
        # `NSTDB_DIR`), so no signature changes are needed.
        import load_real_ecg as _ecg_mod
        import load_real_noise as _noise_mod
        _ecg_mod.DATA_ROOT = data_dir
        _noise_mod.NSTDB_DIR = os.path.join(data_dir, "nstdb")
    ecg_loaded = load_ecg_records(records, channel=0)
    noise_loaded = load_noise_records(NSTDB_NOISE_RECORDS, channel=0)
    if not ecg_loaded or not noise_loaded:
        return None, {"ecg": len(ecg_loaded), "noise": len(noise_loaded),
                      "data_dir": data_dir}

    cells = []
    rng = np.random.default_rng(seed)
    for noise in sorted(noise_loaded, key=lambda d: d["category"]):
        for target in snrs:
            for ecg in ecg_loaded:
                fs_ecg = ecg["fs"]
                fs_noise = noise["fs"]
                n_ecg = round(10.0 * fs_ecg)
                n_noise = round(10.0 * fs_noise)
                for seg in range(segments):
                    ecg_start = seg * n_ecg
                    if ecg_start + n_ecg > len(ecg["signal"]):
                        break
                    clean = ecg["signal"][ecg_start:ecg_start + n_ecg]
                    if len(noise["signal"]) < n_noise:
                        continue
                    noise_seg = noise["signal"][
                        int(rng.integers(0, len(noise["signal"]) - n_noise + 1)):
                    ][:n_noise]
                    try:
                        noisy, achieved = mix_to_snr(
                            clean, noise_seg, float(target),
                            fs_clean=fs_ecg, fs_noise=fs_noise)
                    except ValueError:
                        continue
                    _rp = ecg.get("r_peaks")
                    ref_peaks = np.asarray(_rp if _rp is not None else [],
                                           dtype=int)
                    ref_peaks = ref_peaks[(ref_peaks >= ecg_start) &
                                          (ref_peaks < ecg_start + n_ecg)]
                    ref_peaks = ref_peaks - ecg_start
                    cells.append({
                        "noise_category": noise["category"],
                        "noise_record": noise["name"],
                        "ecg_record": ecg["name"],
                        "segment": seg,
                        "target_snr_db": float(target),
                        "achieved_power_ratio_snr_db": float(achieved),
                        "clean": clean,
                        "noisy": noisy,
                        "fs": float(fs_ecg),
                        "ref_peaks": ref_peaks,
                    })
    return (cells, {"mode": "real", "n_cells": len(cells)})


def _synthetic_cell(seed):
    """Deterministic synthetic ECG + noise pair for offline evaluation."""
    rng = np.random.default_rng(seed)
    fs = 250.0
    n = int(10 * fs)
    t = np.arange(n) / fs
    # QRS-like morphology: sum of Gaussians at ~5 Hz base rate
    clean = np.zeros(n)
    for k in range(10):
        rr = 60.0 / (75.0 + 5 * np.sin(2 * np.pi * (k + 1) / 20.0))
        q = np.exp(-((t - k * rr) % rr) ** 2 / (2 * 0.05 ** 2))
        clean += 0.02 * q
    clean = clean + rng.normal(0, 0.01, n)  # noise-free baseline
    noise = rng.normal(0, 0.1, n)
    return clean, noise, fs


def _synthetic_cells(snrs, segments, seed=DEFAULT_SEED):
    cells = []
    for target in snrs:
        for seg in range(segments):
            clean, noise, fs = _synthetic_cell(
                1000 + seed * 1009 + seg * 7 + int(target))
            # scale noise so the target SNR holds (power-ratio convention)
            sig_p = float(np.mean(clean ** 2))
            noi_p = float(np.mean(noise ** 2))
            if noi_p <= 0:
                continue
            g = np.sqrt(sig_p / (noi_p * 10 ** (target / 10.0)))
            noisy = clean + g * noise
            cells.append({
                "noise_category": "synthetic",
                "noise_record": "synthetic",
                "ecg_record": "synthetic",
                "segment": seg,
                "target_snr_db": float(target),
                "achieved_power_ratio_snr_db": float(10 * np.log10(
                    sig_p / np.mean((g * noise) ** 2))),
                "clean": clean,
                "noisy": noisy,
                "fs": float(fs),
                "ref_peaks": np.array([], dtype=int),
            })
    return cells, {"mode": "synthetic", "n_cells": len(cells)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def zero_mean(x):
    """DC-offset convention shared with the repo: mean removed from both
    sides before waveform-error metrics."""
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def snr_db(reference, estimate):
    """AC (DC-removed) SNR: 10*log10(P_ref / P_est), P = mean(x**2)."""
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    p_ref = float(np.mean(zero_mean(ref) ** 2))
    p_est = float(np.mean(zero_mean(est) ** 2))
    if p_ref <= 0 or p_est <= 0:
        return float("nan")
    return float(10 * np.log10(p_ref / p_est))


def rmse(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((zero_mean(a) - zero_mean(b)) ** 2)))


def nrmse(a, b):
    rng = np.asarray(zero_mean(a), dtype=float).max() - \
        np.asarray(zero_mean(a), dtype=float).min()
    if rng <= 0:
        return float("nan")
    return rmse(a, b) / rng


def prd(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(100 * np.linalg.norm(zero_mean(a) - zero_mean(b)) /
                 np.linalg.norm(zero_mean(a)))


def pearson(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ac, bc = a - a.mean(), b - b.mean()
    den = np.linalg.norm(ac) * np.linalg.norm(bc)
    return float(np.dot(ac, bc) / den) if den > 0 else float("nan")


def rpeak_metrics(ref_peaks, detected_peaks, fs, tolerance_ms=DEFAULT_TOLERANCE_MS):
    """One-to-one matching at the predefined tolerance (±150 ms default).

    Returns sensitivity / precision / F1 / MAE in ms / tp / fp / fn.
    """
    ref = np.asarray(ref_peaks, dtype=int)
    det = np.asarray(detected_peaks, dtype=int)
    tol = round(tolerance_ms / 1000.0 * fs)
    # greedy closest-first one-to-one matching
    cands = []
    for i, r in enumerate(ref):
        for j, d in enumerate(det):
            diff = abs(int(r) - int(d))
            if diff <= tol:
                cands.append((diff, i, j))
    cands.sort()
    used_r, used_d, tp = set(), set(), []
    for _diff, i, j in cands:
        if i not in used_r and j not in used_d:
            used_r.add(i); used_d.add(j); tp.append((i, j))
    n_ref, n_det = ref.size, det.size
    sens = len(tp) / n_ref if n_ref else float("nan")
    prec = len(tp) / n_det if n_det else float("nan")
    f1 = (2 * prec * sens / (prec + sens)
          if np.isfinite(prec) and np.isfinite(sens) and prec + sens > 0
          else float("nan"))
    mae_ms = float(np.mean([abs(ref[i] - det[j]) for i, j in tp]) / fs * 1000.0) \
        if tp else float("nan")
    return {"rpeak_sensitivity": sens, "rpeak_precision": prec,
            "rpeak_f1": f1, "rpeak_mae_ms": mae_ms,
            "rpeak_tp": len(tp), "rpeak_fp": n_det - len(tp),
            "rpeak_fn": n_ref - len(tp)}


def detect_r_peaks_default(x, fs):
    """R-peak detector used when the repo module is unavailable."""
    from scipy.signal import find_peaks
    b = np.asarray(x, dtype=float)
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(3, [5.0, 40.0], btype="band", fs=fs, output="sos")
        f = sosfiltfilt(sos, b)
    except Exception:  # noqa: BLE001
        f = b
    peaks, _ = find_peaks(f, distance=max(1, int(0.24 * fs)),
                          prominence=max(1e-6, 0.3 * float(np.std(f))))
    return peaks


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _digest(x):
    return hashlib.sha256(
        np.ascontiguousarray(np.round(np.asarray(x, dtype=float), 6)).tobytes()
    ).hexdigest()


def _score_structure(out, expected_len):
    """Structure validity: ndarray, 1-D, non-empty, right shape, finite,
    and non-degenerate (a constant output is invalid — there is no
    signal left to denoise)."""
    if not isinstance(out, np.ndarray):
        return False
    if out.ndim != 1 or out.size == 0:
        return False
    if expected_len is not None and out.size != expected_len:
        return False
    if not np.all(np.isfinite(out)):
        return False
    # Degenerate constant output (e.g. zeros or flat values) carries no
    # signal content — reject when the input is long enough to have
    # variation.  Short synthetic inputs can be legitimately flat.
    return not (out.size >= 8 and float(np.ptp(out)) <= 0.0)


def _score_numerical_validity(out):
    if out is None:
        return False
    a = np.asarray(out, dtype=float)
    return bool(np.all(np.isfinite(a)) and a.size > 0)


def _score_signal_processing(noisy, out, fs, solution):
    """Score suppression of **controlled 50 Hz and 60 Hz probe tones**.

    A controlled 5 %-of-input-RMS sinusoid is injected at the *standard*
    mains frequencies (50 Hz and 60 Hz) on top of the given noisy signal,
    and the solution is re-run on the probe-injected input.  The score is
    the attenuation of each probe tone in the output relative to its
    power in the probe-injected input — clipped to [0, 30] dB and
    normalized to [0, 1].  A 0.5–40 Hz band-pass suppresses both probes;
    the raw/wrong-frequency notch leaves them unattenuated (score 0).
    """
    from scipy.fft import rfft, rfftfreq

    noisy = np.asarray(noisy, dtype=float)

    def _bin_pwr(x, f0):
        S = np.abs(rfft(x))
        f = rfftfreq(x.size, d=1 / fs)
        idx = int(np.argmin(np.abs(f - f0)))
        return float(S[idx] ** 2)

    n = min(int(noisy.size), 4000)
    if n < 16:
        return 0.0
    a = 0.05 * float(np.sqrt(np.mean(noisy[:n] ** 2)))
    if not (np.isfinite(a) and a > 0):
        return 0.0
    t = np.arange(n) / fs

    supp = []
    for f0 in (50.0, 60.0):
        probe = a * np.sin(2 * np.pi * f0 * t)
        mixed = noisy[:n] + probe
        try:
            res = solution.denoise(mixed, fs)
            out_p = np.asarray(res.signal, dtype=float)
        except Exception:  # noqa: BLE001
            return 0.0
        in_p = _bin_pwr(mixed, f0)
        out_pwr = _bin_pwr(out_p[:n], f0) if out_p.size >= n else 0.0
        att = -10.0 * np.log10(out_pwr / in_p) if in_p > 0 else 0.0
        supp.append(min(max(att, 0.0), 30.0) / 30.0)
    return float(np.mean(supp))


def _score_performance(cells, out_by_idx, detect_fn, tolerance_ms=DEFAULT_TOLERANCE_MS):
    imps, rmses, nrmss, prds, pears, rpks = [], [], [], [], [], []
    for cell, out in zip(cells, out_by_idx):
        clean = np.asarray(cell["clean"], dtype=float)
        est = np.asarray(out, dtype=float)[:clean.size] if out.size >= clean.size \
            else np.resize(out, clean.size)
        imps.append(snr_db(clean, est) - snr_db(clean, cell["noisy"]))
        rmses.append(rmse(clean, est))
        nrmss.append(nrmse(clean, est))
        prds.append(prd(clean, est))
        pears.append(pearson(clean, est))
        det = detect_fn(est, cell["fs"])
        rpks.append(rpeak_metrics(cell["ref_peaks"], det, cell["fs"],
                                  tolerance_ms=tolerance_ms))
    mean_imp = _mean(imps)
    # Performance: honest SNR improvement mapped into [0, 1].  Improvement
    # in [-10, +30] dB -> [0, 1] (clipped).
    perf = float(np.mean([np.clip((v + 10.0) / 40.0, 0.0, 1.0)
                          for v in imps])) if imps else 0.0
    return perf, {
        "mean_snr_improvement_db": mean_imp,
        "mean_rmse": _mean(rmses),
        "mean_nrmse": _mean(nrmss),
        "mean_prd": _mean(prds),
        "mean_pearson": _mean(pears),
        "mean_rpeak_sensitivity": _mean([r["rpeak_sensitivity"] for r in rpks]),
        "mean_rpeak_precision": _mean([r["rpeak_precision"] for r in rpks]),
        "mean_rpeak_f1": _mean([r["rpeak_f1"] for r in rpks]),
        "n_cells": len(cells),
        "improvements_db": [round(v, 3) for v in imps],
        "per_cell_rpeak": rpks,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(solution, records=None, snrs=None, segments=DEFAULT_SEGMENTS,
             seed=DEFAULT_SEED, tolerance_ms=DEFAULT_TOLERANCE_MS,
             data_dir=None):
    """Run the benchmark on ``solution`` and return the metric dict.

    Parameters
    ----------
    solution : object
        Callable with ``denoise(noisy, fs)`` returning an object with a
        ``signal`` attribute (or a ndarray).
    records : list of str
        ECG records for the real-data mode.
    snrs : list of float
        Target SNR levels.
    segments : int
        Independent noise segments per cell (default 10).
    seed : int
        Seed for the real-data noise window RNG and the synthetic fallback
        generator.
    tolerance_ms : float
        R-peak one-to-one matching tolerance (default 150 ms — the
        predefined value; not chosen from results).
    data_dir : str or None
        PhysioNet data root passed to the repo loaders; ``None`` uses the
        loader's default.
    """
    records = list(records or DEFAULT_RECORDS)
    snrs = list(snrs or TARGET_SNR_LEVELS)

    cells, meta = _load_real_data(records, snrs, segments, seed=seed)
    if data_dir is not None and cells:
        # Re-load honouring the caller's data directory (loaders default
        # when None).
        cells, meta = _load_real_data(
            records, snrs, segments, seed=seed,
            override_dir=data_dir)
    if cells is None or not cells:
        cells, meta = _synthetic_cells(snrs, segments, seed=seed)

    detect_fn = detect_r_peaks_default
    if P2_SRC.is_dir():
        m = _import_optional("evaluate_denoising")
        if m is not None and hasattr(m, "detect_r_peaks"):
            detect_fn = m.detect_r_peaks

    outputs = []
    structure_ok = True
    numerical_valid = True
    sp_scores = []
    for cell in cells:
        res = solution.denoise(cell["noisy"], cell["fs"])
        out = res.signal if hasattr(res, "signal") else res
        out = np.asarray(out, dtype=float)
        ok = _score_structure(out, cell["noisy"].size)
        structure_ok = structure_ok and ok
        numerical_valid = numerical_valid and _score_numerical_validity(out)
        sp_scores.append(_score_signal_processing(
            cell["noisy"], out, cell["fs"], solution))
        outputs.append(out)

    perf, perf_metrics = _score_performance(
        cells, outputs, detect_fn, tolerance_ms=tolerance_ms)
    sp = _mean(sp_scores)

    # Reproducibility: run twice on the first cell and compare digests.
    first_cell = cells[0] if cells else None
    repro_ok = True
    if first_cell is not None:
        r1 = np.asarray(solution.denoise(first_cell["noisy"], first_cell["fs"]).signal,
                        dtype=float)
        r2 = np.asarray(solution.denoise(first_cell["noisy"], first_cell["fs"]).signal,
                        dtype=float)
        repro_ok = bool(_digest(r1) == _digest(r2))

    # Edge robustness (audit 2.3.2): the solution must not crash and must
    # return a finite output of the correct shape on a short segment and on
    # zero-power input.  (A degenerate constant output is acceptable on
    # degenerate input — what matters is shape/finiteness, not SNR there.)
    edge_ok = True
    try:
        if cells:
            short = np.zeros(16, dtype=float)
            o = solution.denoise(short, 250.0)
            s = np.asarray(o.signal, dtype=float)
            edge_ok = edge_ok and (s.size == short.size
                                   and np.all(np.isfinite(s)))
            zero = np.zeros(int(5 * cells[0]["fs"]), dtype=float)
            o = solution.denoise(zero, cells[0]["fs"])
            z = np.asarray(o.signal, dtype=float)
            edge_ok = edge_ok and (z.size == zero.size
                                   and np.all(np.isfinite(z)))
    except Exception:  # noqa: BLE001
        edge_ok = False

    # Overall weighted score (scaled to [0, 1] each).
    overall = (
        SCORING_RUBRIC["structure"] * (1.0 if structure_ok else 0.0) +
        SCORING_RUBRIC["numerical_validity"] * (1.0 if numerical_valid else 0.0) +
        SCORING_RUBRIC["signal_processing"] * sp +
        SCORING_RUBRIC["performance"] * perf +
        SCORING_RUBRIC["reproducibility"] * (1.0 if repro_ok else 0.0) +
        SCORING_RUBRIC["edge_robustness"] * (1.0 if edge_ok else 0.0)
    )

    return {
        "mode": meta["mode"],
        "records": records,
        "n_cells": len(cells),
        "snrs": snrs,
        "segments_per_cell": segments,
        "seed": seed,
        "peak_tolerance_ms": tolerance_ms,
        "overall_score": round(float(overall), 4),
        "sub_scores": {
            "structure": 1.0 if structure_ok else 0.0,
            "numerical_validity": 1.0 if numerical_valid else 0.0,
            "signal_processing": round(float(sp), 4),
            "performance": round(float(perf), 4),
            "reproducibility": 1.0 if repro_ok else 0.0,
            "edge_robustness": 1.0 if edge_ok else 0.0,
        },
        "performance_metrics": {k: (round(v, 4) if isinstance(v, float)
                                    and np.isfinite(v) else v)
                                for k, v in perf_metrics.items()},
        "reproducibility_ok": bool(repro_ok),
        "edge_robustness_ok": bool(edge_ok),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Benchmark evaluator")
    p.add_argument("--solution", default="reference_solution",
                   choices=["reference_solution", "starter_solution"])
    p.add_argument("--records", default=None)
    p.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    a = p.parse_args()
    mod = __import__(a.solution)
    sol = getattr(mod, a.solution)()
    records = [r for r in a.records.split(",")] if a.records else None
    result = evaluate(sol, records=records, segments=a.segments)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()

