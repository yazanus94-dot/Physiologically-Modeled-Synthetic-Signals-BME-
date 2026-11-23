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
#: Segment duration (seconds) of every evaluation cell — one documented
#: value used by the evaluator, TASK.md and expected_results.json.
DEFAULT_SEGMENT_S = 10.0
#: Worst improvement the performance mapping accepts; also the score given
#: to a cell whose output is malformed (maps to 0 after clipping).
PERF_FLOOR_DB = -10.0
#: Best improvement the performance mapping can use; also the value given
#: to a mathematically perfect reconstruction (zero error power), whose
#: SNR improvement is ``+inf``.
PERF_CEIL_DB = 30.0


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


def _software_versions():
    """Software versions stamped onto results (audit 2.4).

    Uses the repo's shared ``versions.py`` when it is importable and
    degrades to a minimal record when the package is run standalone.
    """
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from versions import get_versions
        return get_versions()
    except Exception:  # noqa: BLE001 - standalone copy of the package
        import platform
        import sys as _sys
        return {"python": _sys.version.split()[0],
                "platform": platform.platform()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_real_data(records, snrs, segments, seed=DEFAULT_SEED,
                    override_dir=None):
    """Load real ECG + noise. Returns ``(cells, loader_info)``.

    ``override_dir`` is the **PhysioNet data root** the caller asked for
    (the directory that contains ``mitdb/`` and ``nstdb/``).  ``None``
    keeps the repo loaders' own default (``<repo>/data/physionet``).

    Audit 2.3.5: the override is applied to the loaders' module-level
    directory constants *for the duration of this call only* and is
    restored afterwards, so a caller-supplied ``data_dir`` really does
    select the data that is read and cannot leak into later calls.
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

    # The loaders derive their paths from module-level constants at call
    # time (`records_dir()` reads `DATA_ROOT`, `load_noise_record()` reads
    # `NSTDB_DIR`), so pointing those constants at the requested root is
    # enough — no signature changes are needed.
    saved_dirs = None
    if override_dir:
        import load_real_ecg as _ecg_mod
        import load_real_noise as _noise_mod
        saved_dirs = (_ecg_mod.DATA_ROOT, _noise_mod.NSTDB_DIR)
        _ecg_mod.DATA_ROOT = str(override_dir)
        _noise_mod.NSTDB_DIR = os.path.join(str(override_dir), "nstdb")

    try:
        ecg_loaded = load_ecg_records(records, channel=0)
        noise_loaded = load_noise_records(NSTDB_NOISE_RECORDS, channel=0)
    finally:
        if saved_dirs is not None:
            _ecg_mod.DATA_ROOT, _noise_mod.NSTDB_DIR = saved_dirs

    if not ecg_loaded or not noise_loaded:
        return None, {"ecg": len(ecg_loaded), "noise": len(noise_loaded),
                      "data_dir": str(override_dir) if override_dir else None}

    cells = []
    rng = np.random.default_rng(seed)
    for noise in sorted(noise_loaded, key=lambda d: d["category"]):
        for target in snrs:
            for ecg in ecg_loaded:
                fs_ecg = ecg["fs"]
                fs_noise = noise["fs"]
                n_ecg = round(DEFAULT_SEGMENT_S * fs_ecg)
                n_noise = round(DEFAULT_SEGMENT_S * fs_noise)
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
    return (cells, {"mode": "real", "n_cells": len(cells),
                    "data_dir": str(override_dir) if override_dir else None})


def _synthetic_cell(seed):
    """Deterministic synthetic ECG + noise pair for offline evaluation.

    One P-QRS-T-like complex per beat at ~75 bpm with respiratory sinus
    arrhythmia (the rate varies as a slow sinusoid in the beat index), plus
    a low-level measurement noise floor.

    Audit 2.5: the previous generator built each beat as
    ``exp(-((t - k*rr) % rr)**2 / ...)``, i.e. a *spike train at multiples
    of rr* rather than a single complex at ``k*rr``.  Summing ten such
    trains produced ~12 beats per second (≈720 bpm) whose energy sat above
    the 40 Hz diagnostic band, so any sensible band-limiting denoiser
    looked worse than its input on the offline fallback.  The fallback now
    generates a physiologically plausible rhythm, which is what the
    docstring always claimed.
    """
    rng = np.random.default_rng(seed)
    fs = 250.0
    n = int(DEFAULT_SEGMENT_S * fs)
    t = np.arange(n) / fs
    clean = np.zeros(n)
    # Start after the window edge and stop before it, so the segment begins
    # and ends in baseline: a beat truncated by the segment boundary is an
    # artifact no denoiser can recover and would dominate the offline score.
    beat_t, k = 0.3, 0
    while beat_t < t[-1] - 0.5:
        rr = 60.0 / (75.0 + 5 * np.sin(2 * np.pi * (k + 1) / 20.0))
        # Narrow QRS-like spike + broad low-amplitude T wave.
        clean += 0.9 * np.exp(-((t - beat_t) ** 2) / (2 * 0.012 ** 2))
        clean += 0.15 * np.exp(-((t - beat_t - 0.25) ** 2) / (2 * 0.05 ** 2))
        beat_t += rr
        k += 1
    clean = clean + rng.normal(0, 0.01, n)  # measurement noise floor
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


def unwrap_output(result):
    """Extract the signal array from a solution's ``denoise`` return value.

    The documented contract is an object with a ``signal`` attribute, but a
    bare ``ndarray`` (or any array-like) is accepted too.  **Every** path
    that inspects solution output goes through this helper: the main
    evaluation loop, the 50/60 Hz probe scoring, the edge-robustness checks
    and the reproducibility digest.

    Regression (audit 2.6): the probe and edge checks used to read
    ``result.signal`` directly, so a bare-ndarray solution raised
    ``AttributeError`` inside both; the exception was swallowed and the
    solution scored ``signal_processing = 0.0`` and
    ``edge_robustness = 0.0`` — 0.35 of the rubric lost purely for its
    return type, with a valid waveform in hand.
    """
    return np.asarray(getattr(result, "signal", result), dtype=float)


def _as_1d(x):
    """Coerce ``x`` to a non-empty 1-D float array, else return ``None``."""
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 1 or a.size == 0:
        return None
    return a


def snr_db(reference, estimate):
    """AC (DC-removed) **reconstruction** SNR in dB.

    ``10 * log10(P_ref / P_err)`` where

    * ``P_ref = mean(zero_mean(reference) ** 2)`` is the signal power,
    * ``P_err = mean((zero_mean(reference) - zero_mean(estimate)) ** 2)``
      is the **error** power (the residual the denoiser left behind).

    Audit 2.5: this function used to divide by the power of the *estimate*
    instead of the error power.  That is not an SNR — under that definition
    ``improvement = 10*log10(P_noisy / P_estimate)``, so a solution that
    simply scaled its input down scored as if it had denoised it (scaling
    by 0.1 earned +20 dB while removing no noise at all).  The error-based
    definition is the standard one and is also what Project 2 uses
    (``project-2-ecg-denoising/src/denoise_methods.py::snr_db``).

    Returns ``+inf`` for a perfect reconstruction (zero error power) and
    ``nan`` whenever the value is undefined — including empty input,
    shape-mismatched input, non-1-D input (the benchmark's contract is a
    1-D segment), and NaN/Inf samples.  Returning ``nan`` rather than
    raising keeps the caller's finite-score mapping in charge.
    """
    ref = _as_1d(reference)
    est = _as_1d(estimate)
    if ref is None or est is None or ref.size != est.size:
        return float("nan")
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(est))):
        # Undefined: a single NaN/Inf sample has no meaningful error power
        # (and ``inf - inf`` would only emit a numpy warning).
        return float("nan")
    ref_c = zero_mean(ref)
    est_c = zero_mean(est)
    p_ref = float(np.mean(ref_c ** 2))
    p_err = float(np.mean((ref_c - est_c) ** 2))
    if not (np.isfinite(p_ref) and np.isfinite(p_err)) or p_ref <= 0:
        return float("nan")
    if p_err <= 0:
        return float("inf")
    return float(10 * np.log10(p_ref / p_err))


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
    den = np.linalg.norm(zero_mean(a))
    if den <= 0:
        return float("nan")   # degenerate reference (zero power)
    return float(100 * np.linalg.norm(zero_mean(a) - zero_mean(b)) / den)


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


#: An output must retain at least this fraction of the input's peak-to-peak
#: range.  Guards against "almost constant" outputs that are technically
#: non-flat but carry no signal (audit 2.3.8).
MIN_PTP_RETENTION = 1e-6


def _score_structure(out, expected_len, expected_ptp=None):
    """Structure validity: ndarray, 1-D, non-empty, right shape, finite,
    and non-degenerate (a constant output is invalid — there is no
    signal left to denoise).

    ``expected_ptp`` is the input's peak-to-peak range.  When given, the
    output must retain at least ``MIN_PTP_RETENTION`` of it, so a nearly
    flat array cannot pass as a denoised signal (audit 2.3.1 / 2.3.8).
    """
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
    if out.size < 8:
        return True
    ptp = float(np.ptp(out))
    if ptp <= 0.0:
        return False
    if expected_ptp is not None and float(expected_ptp) > 0.0:
        return ptp >= MIN_PTP_RETENTION * float(expected_ptp)
    return True


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
            out_p = unwrap_output(res)
        except Exception:  # noqa: BLE001
            return 0.0
        in_p = _bin_pwr(mixed, f0)
        out_pwr = _bin_pwr(out_p[:n], f0) if out_p.size >= n else 0.0
        if in_p > 0 and out_pwr > 0:
            att = -10.0 * np.log10(out_pwr / in_p)
        elif in_p > 0:
            # No probe energy left in the output at all -> the full cap.
            att = 30.0
        else:
            att = 0.0
        supp.append(min(max(att, 0.0), 30.0) / 30.0)
    return float(np.mean(supp))


def _score_performance(cells, out_by_idx, detect_fn,
                       tolerance_ms=DEFAULT_TOLERANCE_MS, valid=None):
    """Aggregate the performance sub-score and its diagnostic metrics.

    Audit 2.3.8: a cell whose output failed the structure check earns **no**
    performance credit — it is scored at ``PERF_FLOOR_DB`` (which maps to
    0 after clipping) instead of being silently tiled/resized into a
    plausible-looking waveform.  Malformed outputs therefore cannot bank a
    high performance sub-score while only losing the structure sub-score.

    ``valid[i]`` is the structure verdict for cell ``i`` (default: all
    valid, so direct callers of this helper keep working).
    """
    if valid is None:
        valid = [True] * len(cells)
    imps, scored, rmses, nrmss, prds, pears, rpks = [], [], [], [], [], [], []
    for cell, out, is_valid in zip(cells, out_by_idx, valid):
        clean = np.asarray(cell["clean"], dtype=float)
        out = np.asarray(out, dtype=float)
        if not is_valid or out.size != clean.size:
            # Malformed output: floor the score, do not fabricate a signal
            # by tiling it to the reference length.
            scored.append(PERF_FLOOR_DB)
            continue

        # Performance credit (floored when the improvement is undefined,
        # e.g. a degenerate zero-power reference cell).
        imp = snr_db(clean, out) - snr_db(clean, cell["noisy"])
        if np.isfinite(imp):
            imps.append(imp)
            scored.append(imp)
        elif np.isposinf(imp):
            # Perfect reconstruction (zero error power): full credit, and
            # a finite diagnostic instead of an unrepresentable ``inf``.
            imps.append(PERF_CEIL_DB)
            scored.append(PERF_CEIL_DB)
        else:
            scored.append(PERF_FLOOR_DB)

        # Diagnostics are reported for every structurally valid output,
        # independently of whether its SNR improvement is defined.
        rmses.append(rmse(clean, out))
        nrmss.append(nrmse(clean, out))
        prds.append(prd(clean, out))
        pears.append(pearson(clean, out))
        det = detect_fn(out, cell["fs"])
        rpks.append(rpeak_metrics(cell["ref_peaks"], det, cell["fs"],
                                  tolerance_ms=tolerance_ms))
    mean_imp = _mean(imps)
    # Performance: honest error-reduction ("denoising"), mapped into [0, 1].
    # Improvement in [PERF_FLOOR_DB, PERF_CEIL_DB] = [-10, +30] dB -> [0, 1].
    span = PERF_CEIL_DB - PERF_FLOOR_DB
    perf = float(np.mean([np.clip((v - PERF_FLOOR_DB) / span, 0.0, 1.0)
                          for v in scored])) if scored else 0.0
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
        "n_scored_cells": len(scored),
        "n_well_defined_cells": len(imps),
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
        PhysioNet **data root** (the directory containing ``mitdb/`` and
        ``nstdb/``) passed to the repo loaders.  It is honoured on the
        first load, not only when the default directory already holds
        data.  ``None`` uses the loaders' default (``<repo>/data/physionet``).

    Notes
    -----
    This is a **fixed, independent evaluation protocol**: there is no
    training step, so no train/validation/test split exists (see TASK.md
    section B).
    """
    records = list(records or DEFAULT_RECORDS)
    snrs = list(snrs or TARGET_SNR_LEVELS)

    # Audit 2.3.5: one call, honouring ``data_dir`` from the start — the
    # previous implementation only re-loaded with the caller's directory
    # when the *default* directory already had data, so a non-default
    # ``data_dir`` was silently ignored on a fresh checkout.
    cells, meta = _load_real_data(records, snrs, segments, seed=seed,
                                  override_dir=data_dir)
    if cells is None or not cells:
        cells, meta = _synthetic_cells(snrs, segments, seed=seed)
        if data_dir is not None:
            meta = dict(meta, requested_data_dir=str(data_dir),
                        data_unavailable=True)

    detect_fn = detect_r_peaks_default
    if P2_SRC.is_dir():
        m = _import_optional("evaluate_denoising")
        if m is not None and hasattr(m, "detect_r_peaks"):
            detect_fn = m.detect_r_peaks

    outputs = []
    valid_flags = []
    structure_ok = True
    numerical_valid = True
    sp_scores = []
    for cell in cells:
        res = solution.denoise(cell["noisy"], cell["fs"])
        out = unwrap_output(res)
        # ``expected_ptp`` makes the structure check reject flat/near-flat
        # outputs, which is what lets us withhold performance and
        # signal-processing credit from them (audit 2.3.8).
        ok = _score_structure(out, cell["noisy"].size,
                              expected_ptp=float(np.ptp(cell["noisy"])))
        structure_ok = structure_ok and ok
        numerical_valid = numerical_valid and _score_numerical_validity(out)
        # A structurally invalid output earns no signal-processing credit:
        # a flat array has (numerically) zero power at 50/60 Hz and would
        # otherwise look like a perfect mains notch (audit 2.3.8).
        sp_scores.append(_score_signal_processing(
            cell["noisy"], out, cell["fs"], solution) if ok else 0.0)
        outputs.append(out)
        valid_flags.append(bool(ok))

    perf, perf_metrics = _score_performance(
        cells, outputs, detect_fn, tolerance_ms=tolerance_ms,
        valid=valid_flags)
    sp = _mean(sp_scores)
    if not np.isfinite(sp):
        sp = 0.0

    # Reproducibility: run twice on the first cell and compare digests.
    # (Uses the same shared ``unwrap_output`` as every other path, so a
    # plain-array solution cannot crash here, and exposes the digest so
    # TASK.md section A's promise holds.)
    first_cell = cells[0] if cells else None
    repro_ok = True
    repro_digest = None
    if first_cell is not None:
        r1 = unwrap_output(
            solution.denoise(first_cell["noisy"], first_cell["fs"]))
        r2 = unwrap_output(
            solution.denoise(first_cell["noisy"], first_cell["fs"]))
        repro_digest = _digest(r1)
        repro_ok = bool(repro_digest == _digest(r2))

    # Edge robustness (audit 2.3.2): the solution must not crash and must
    # return a finite output of the correct shape on a short segment and on
    # zero-power input.  (A degenerate constant output is acceptable on
    # degenerate input — what matters is shape/finiteness, not SNR there.)
    # Uses ``unwrap_output`` so a bare-ndarray solution is judged on what it
    # returned, not on how it returned it (audit 2.6).
    edge_ok = True
    try:
        if cells:
            short = np.zeros(16, dtype=float)
            s = unwrap_output(solution.denoise(short, 250.0))
            edge_ok = edge_ok and (s.size == short.size
                                   and np.all(np.isfinite(s)))
            zero = np.zeros(int(5 * cells[0]["fs"]), dtype=float)
            z = unwrap_output(solution.denoise(zero, cells[0]["fs"]))
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
        "noise_records": list(NSTDB_NOISE_RECORDS),
        "n_cells": len(cells),
        "snrs": snrs,
        "segments_per_cell": segments,
        "segment_duration_s": DEFAULT_SEGMENT_S,
        "seed": seed,
        "peak_tolerance_ms": tolerance_ms,
        "data_dir": (str(data_dir) if data_dir is not None
                     else meta.get("data_dir")),
        "overall_score": round(float(overall), 4),
        "structure_ok": bool(structure_ok),
        "reproducibility_digest": repro_digest,
        "provenance": {
            "protocol": "fixed independent evaluation (no train/val/test split)",
            "ecg_records": records,
            "noise_records": list(NSTDB_NOISE_RECORDS),
            "noise_categories": ["baseline_wander", "muscle_artifact",
                                 "electrode_motion"],
            "target_snr_db": snrs,
            "segments_per_cell": segments,
            "segment_duration_s": DEFAULT_SEGMENT_S,
            "n_cells": len(cells),
            "seed": seed,
            "peak_tolerance_ms": tolerance_ms,
            "data_dir": (str(data_dir) if data_dir is not None
                         else meta.get("data_dir")),
            "mode": meta["mode"],
            "software": _software_versions(),
            # Present only when the requested directory could not be used.
            **({k: meta[k] for k in ("requested_data_dir", "data_unavailable")
                if k in meta}),
        },
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

