# PhysioNet-style Benchmark Task — MIT-BIH + NSTDB ECG Denoising

*Independent portfolio benchmark inspired by PhysioNet-style scientific
evaluation — **not** an official PhysioNet Challenge submission.*

---

## A. Problem statement

Given **clean real ECG** (a segment of a PhysioNet MIT-BIH record) mixed
with **real recorded noise** from the PhysioNet NSTDB (baseline wander
`bw`, muscle artifact `ma`, electrode motion `em`), the task is to recover
the clean signal from the mixture as well as possible.

- **Input:** a noisy ECG segment (NumPy `ndarray`, `float`, one-dimensional,
  same length as the reference).
- **Expected output:** a denoised ECG segment (NumPy `ndarray`, `float`,
  one-dimensional, same shape and length as the input).
- **Constraints:**
  1. The output must be **finite** (no NaN/Inf).
  2. The output must have the **same shape** as the input.
  3. Do not use the clean reference to *generate* features or to tune
     hyperparameters on the final test set (evaluation is external).
  4. The denoiser must be a **causal or real-time-friendly** implementation
     (no look-ahead beyond the segment under denoising) — but within the
     segment, zero-phase filtering is allowed since the segment is the unit
     of processing.
- **Evaluation criteria:** the evaluator in `evaluator.py` computes, per
  (noise category × target SNR):
  - **error-based** SNR improvement over the raw noisy signal (dB):
    `10*log10(P_signal / P_error)` before versus after denoising, where
    `P_error` is the power of `reference - estimate`.  Audit 2.5: the
    earlier definition divided by the *estimate's* power, which rewarded
    attenuation instead of denoising (scaling the input by 0.1 scored
    +20 dB while removing no noise); it now matches Project 2's
    `snr_db` convention,
  - RMSE / NRMSE between reference and estimate,
  - Pearson correlation,
  - and a **reproducibility digest** of the output array (shape + hash of
    the rounded values) for determinism checks.
  The evaluator returns a single weighted score (see `SCORING_RUBRIC`) plus
  the full metric table.

## B. Dataset and experimental protocol

- **PhysioNet source:** MIT-BIH Arrhythmia Database
  ([PhysioNet — MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/)),
  records **118** and **119** (real, primary clean-ECG records), and the
  MIT-BIH Noise Stress Test Database
  ([PhysioNet — NSTDB v1.0.0](https://physionet.org/content/nstdb/1.0.0/))
  for the three pure-noise records `bw` (baseline wander), `ma` (muscle
  artifact), `em` (electrode motion artifact).
- **Sampling frequency:** read from the WFDB record header
  (`wfdb.rdrecord(path).fs`) — never hardcoded; MIT-BIH happens to be
  360 Hz at the native rate used by this benchmark.
- **Annotation format:** MIT-BIH `.atr` beat annotations
  (`wfdb.rdann(path, "atr")`) are used **only to locate R-peaks for the
  evaluation of R-peak metrics** — they are *not* training features.
- **Target SNR levels:** `[-6, 0, 6, 12, 18, 24]` dB.
- **Train / validation / test separation:** **there is none — this is a
  fixed, independent evaluation protocol.**  The evaluator scores a
  solution against the *same* clean reference signal on every cell
  (3 noise × 6 SNR); there is no training step to split, so no
  train/validation/test cell structure is defined or used.  (A
  supervised benchmark would need a training set; this task is a
  denoising-eval task and its specification matches the evaluator.)
- **Segment duration:** **10 seconds** (`DEFAULT_SEGMENT_S = 10.0`) —
  one single value, used by the evaluator, this document and
  `expected_results.json`.  Every metric in a cell is computed on a
  10 s segment.
- **Segments and trials:** **10 independent noise segments per
  (noise × SNR) cell** (`DEFAULT_SEGMENTS = 10`), giving
  3 noise × 6 SNR × 2 records × 10 segments = **360 cells**.  The ECG
  window starts at `seg * n_ecg` inside each record, and each segment
  draws its own noise window from `np.random.default_rng(seed)` advanced
  sequentially over the (noise → SNR → record → segment) loop — so
  `seed` (default `3`) fully determines which windows are used.
- **Data directory:** `evaluate(..., data_dir=...)` takes the PhysioNet
  **data root** (the directory containing `mitdb/` and `nstdb/`); the
  default `None` uses the repo loaders' own
  `<repo>/data/physionet`.  The override is applied for the duration of
  the load and restored afterwards.
- **One-to-one peak matching** at **±150 ms**
  (`DEFAULT_TOLERANCE_MS = 150.0`, the predefined, never-tuned
  tolerance; `evaluate(..., tolerance_ms=...)`).
- **Signal-processing scoring:** controlled **50 Hz and 60 Hz** mains
  probe tones are injected onto the input (5 % of the input RMS) and the
  solution is re-run on the probe-injected input; the score is the mean
  attenuation (clipped to [0, 30] dB, normalized to [0, 1]).
- **R-peak F1 is a diagnostic, not a scored metric:** it is reported in
  `performance_metrics` (`mean_rpeak_f1`) and the evaluator's metric
  list, but not weighted in `SCORING_RUBRIC`.  The six rubric criteria
  are: structure (15 %), numerical validity (10 %), signal processing
  (25 %), performance (25 %), reproducibility (15 %), edge robustness
  (10 %).
- **DC-offset convention:** waveform-error metrics (`rmse`, `nrmse`, `prd`,
  `snr_db`) are computed after removing the mean from both signals
  (`zero_mean`) — documented in the reference solution.  `snr_db` uses
  the **error** power (`reference - estimate`) as its denominator.

## C. Starter baseline

A straightforward baseline (`starter_solution.py`):

1. Bandpass filter (0.5–40 Hz) on the noisy segment (zero-phase
   `sosfiltfilt`, order 3).
2. Return the filtered signal.

The starter scores modestly but is easy to understand; the reference
solution out-performs it by adding a **median baseline-removal** step and
**50/60 Hz notches**, and by ending the chain in a **40 Hz low-pass**
instead of a 0.5 Hz high-pass (the measurement below explains why):

| | reference | starter |
|---|---:|---:|
| `overall_score` | **0.8366** | 0.6469 |
| mean SNR improvement | **+3.89 dB** | +2.99 dB |
| signal-processing sub-score | **0.9990** | 0.6609 |
| edge robustness | **1.0** | 0.0 (raises on the 16-sample probe) |

Per target SNR (mean error-based SNR improvement, dB):

| target SNR | −6 | 0 | +6 | +12 | +18 | +24 |
|---|---:|---:|---:|---:|---:|---:|
| reference | +10.74 | +8.87 | +6.82 | +3.76 | −0.75 | −6.09 |
| starter | +8.96 | +8.19 | +6.02 | +2.94 | −1.49 | −6.69 |

Measured with `evaluator.evaluate(..., segments=10)` on the real data
(see `expected_results.json`); the offline synthetic fallback reproduces
the same ordering (**+0.168** score, **+1.44 dB** at `segments=10`).

Negative mean improvements at +18/+24 dB are expected and honest: at
those operating points the clean record's own noise floor dominates the
measurement, so there is little error left to reduce.  The starter's
short-segment failure is reproducible: `sosfiltfilt` needs a longer pad
than the 16-sample edge probe provides.

**Why the 0.5 Hz high-pass corner was removed (audit 2.5).**  The clean
reference is the raw MIT-BIH record (recorded through a 0.1 Hz hardware
high-pass), so it still carries real low-frequency ECG content.  Running
the reference chain on the **clean** signal — the best fidelity any
denoiser built from that chain could reach — gives **9.04 dB** with the
0.5–40 Hz band-pass but **11.38 dB** with the 40 Hz low-pass.  The
high-pass corner was deleting signal, not noise; under the old
(estimate-power) SNR definition that loss was invisible.

## D. Reference solution

`reference_solution.py`: a complete, documented implementation that
mirrors the chain of Project 2 (`project-2-ecg-denoising`) — median
baseline removal → mains notch → band limiting — with the band limiter
being a 40 Hz low-pass (see section C for the measurement that justifies
dropping the 0.5 Hz high-pass corner).  It includes:

- `bandpass_filter` (Butterworth 0.5–40 Hz, order 3, zero-phase via
  `sosfiltfilt`) — kept as a documented helper and as Project 2's band
  limiter, but **not** the stage the benchmark reference submits,
- `lowpass_filter` (Butterworth 40 Hz, order 3, zero-phase),
- `median_filter_baseline_removal` (median window 0.6 s),
- `notch_filter` (IIR notch, default 50 Hz; the chain applies both 50 Hz
  and 60 Hz, skipping any that would sit at or above Nyquist),
- `fft_soft_threshold` (FFT-domain soft-threshold: out-of-band bins
  zeroed, in-band magnitudes shrunk by `0.08 * max|magnitude|` with the
  phase preserved),
- `combined_pipeline` (median → 50/60 Hz notches → 40 Hz low-pass).

**What the reference actually runs.**  `denoise(noisy, fs)` **is**
`combined_pipeline(noisy, fs)` — median baseline removal → 50 Hz and
60 Hz notches → 40 Hz low-pass.  `fft_soft_threshold` is provided as a
documented, tested *alternative* stage (it is exercised by the module's
own helpers and is what Project 2's `fft` method uses), but it is **not**
part of the default pipeline: on this protocol it scores below the
submitted chain (measured overall 0.8229 vs 0.8366) — it removes mains
perfectly but its performance sub-score is much lower (0.2918 vs 0.3473)
because zeroing out-of-band bins is a fixed, non-adaptive rule.

Each stage is documented with its parameter choices and the scientific
rationale.  The reference implementation returns a `DenoisingResult`
dataclass (`signal`, `method`) and is deterministic for a fixed input.

## E. Automated evaluator

`evaluator.py` exposes:

```python
def evaluate(solution, records=None, snrs=None, segments=10, seed=3,
             tolerance_ms=150.0, data_dir=None) -> dict
```

Defaults are `records=("118", "119")`, `snrs=(-6, 0, 6, 12, 18, 24)` dB.
`data_dir=None` uses the repo loaders' default `<repo>/data/physionet`;
pass a PhysioNet **data root** to read from somewhere else.

Returns a dict with:

- `overall_score` — the weighted score (see `SCORING_RUBRIC`),
- `structure_ok` — boolean: every output was finite, 1-D, the same
  shape as its input and non-degenerate (a constant/near-flat output
  fails),
- `sub_scores` — the six rubric sub-scores in [0, 1]
  (`structure`, `numerical_validity`, `signal_processing`,
  `performance`, `reproducibility`, `edge_robustness`),
- `performance_metrics` — diagnostics: `mean_snr_improvement_db`
  (**error-based** improvement),
  `mean_rmse`, `mean_nrmse`, `mean_prd`, `mean_pearson`,
  `mean_rpeak_sensitivity` / `mean_rpeak_precision` / `mean_rpeak_f1`
  (F1 is **diagnostic only** — it is not in `SCORING_RUBRIC`),
  `improvements_db`, `per_cell_rpeak`, `n_cells`, `n_scored_cells`,
  `n_well_defined_cells`,
- `provenance` — the fixed protocol (records, noise categories, SNRs,
  segments, segment duration, seed, tolerance, mode) plus the installed
  software versions, so a result can be reproduced from the JSON alone,
- `reproducibility_digest` — SHA-256 hex digest of the rounded output
  of the first cell, identical across runs of a deterministic solution;
- `mode` — `"real"` when PhysioNet data was loaded, `"synthetic"`
  otherwise; plus `reproducibility_ok`, `edge_robustness_ok` and the
  protocol echo (`records`, `snrs`, `segments_per_cell`,
  `segment_duration_s`, `seed`, `peak_tolerance_ms`, `data_dir`).

A **malformed output** (wrong shape, NaN, constant/near-flat) earns no
`performance` and no `signal_processing` credit: it is floored at
`PERF_FLOOR_DB = -10 dB` (0 after clipping) instead of being tiled into
a plausible waveform, and its signal-processing score is 0 (a flat array
has zero power at the probe frequency and would otherwise look like a
perfect mains notch).

The evaluator **loads the PhysioNet data lazily** via
`project-2-ecg-denoising/src/load_real_ecg.py` and
`load_real_noise.py` (or, if those are unavailable, falls back to bundled
synthetic reference data so the evaluator is runnable offline — the task
tests do this too).

## F. Tests and failure cases

`tests/test_solutions.py` (**32 test functions, 34 collected cases**)
contains:

- **Success cases:** `reference_solution` on the evaluator → structure
  ok, finite outputs, `overall_score` within 0.05 of
  `expected_results.json` (protocol fields asserted in *both* modes, so
  CI without data still catches spec drift); `reference` beats
  `starter` by a measured margin (**> +0.10** overall, **> +0.12 dB**
  mean SNR improvement — measured +0.1948 / +1.73 dB on real data at
  `segments=2`, and +0.1682 / +1.47 dB on the offline fallback);
  `reference_solution` is deterministic.
- **Failure cases:**
  - Solution that returns a **constant** signal → `structure_ok` False,
    performance 0, signal-processing 0, **finite** overall score
    (regression: it used to be `NaN`).
  - Solution that returns **NaN/Inf** → `structure_ok` False.
  - Solution with **wrong shape** → `structure_ok` False *and*
    performance 0 (regression: `np.resize` used to tile the malformed
    array and bank 0.24 performance).
  - **Near-flat** output (ptp below `1e-6` of the input's) → rejected.
  - Reproducibility: two runs give identical results.
- **Protocol/evaluator regressions:** `data_dir` honoured when the
  default directory is empty; loader directory constants restored
  after an override; `seed` reaches the real-data loader; `tolerance_ms`
  reaches `rpeak_metrics` (checked with a 168 ms offset that is a miss at
  150 ms and a match at 200 ms); F1 stays diagnostic-only; provenance
  block complete.
- **Metric-definition regressions (audit 2.5):** `snr_db` is error-based,
  so attenuating a high-SNR input loses SNR instead of scoring the old
  `-20*log10(scale)` reward; the offline synthetic generator produces a
  physiological rate (30–180 bpm) with < 5 % of its energy above 40 Hz.
- **Edge cases:** very short segment (16 samples), zero-power input
  (graceful, no crash), short/zero inputs to the reference solution.

## G. Scoring rubric

| Criterion | Weight | Description |
|---|---:|---|
| Output structure | 15% | finite, 1-D, same shape, non-degenerate, `structure_ok=True` |
| Numerical validity | 10% | no NaN/Inf, no shape mismatch, sensible values |
| Signal-processing correctness | 25% | notches + low-pass attenuate controlled 50 Hz and 60 Hz probe tones; morphology preserved; output band-limited to the diagnostic 40 Hz |
| Performance metrics | 25% | average **error-based** SNR improvement, RMSE, NRMSE, PRD across cells — higher is better (scored by improving the mean) |
| Reproducibility | 15% | deterministic output for fixed seed |
| Edge-case robustness | 10% | graceful handling of short segments, zero-power inputs, constant signals |

`overall_score` = weighted average of the sub-scores in [0, 1].

## H. Expected results

`expected_results.json` — generated **by this evaluator** from
`reference_solution.py` on the real PhysioNet data (MIT-BIH records
118/119 + NSTDB `bw`/`ma`/`em`, 10 segments per cell, 10 s segments,
seed 3, ±150 ms tolerance — 360 cells).  It records the full result dict
including the `provenance` block (protocol + software versions).
Reproduce with:

```bash
# from the repo root, with data/physionet/{mitdb,nstdb} present
python -c "import sys,json; sys.path.insert(0,'benchmark-tasks/mitbih_nstdb_denoising_benchmark'); \
from evaluator import evaluate; from reference_solution import reference_solution; \
print(json.dumps(evaluate(reference_solution()), indent=2))"
python -m pytest benchmark-tasks/mitbih_nstdb_denoising_benchmark/tests
```

Key reference values (means over 360 cells, reference pipeline
median → 50/60 Hz notches → 40 Hz low-pass):

- **`overall_score` = 0.8366** (structure 1.0, numerical validity 1.0,
  signal processing 0.9990, performance 0.3473, reproducibility 1.0,
  edge robustness 1.0).
- Mean error-based SNR improvement **+3.89 dB** vs **+2.99 dB** for the
  starter; the reference is strictly better at *every* target SNR from
  −6 dB to +24 dB (see the per-SNR table in section C).
- Starter `overall_score` = 0.6469 (edge robustness 0.0: `sosfiltfilt`
  raises on the 16-sample edge probe) → margin **+0.1897**.
- Offline synthetic fallback reproduces the same ordering with a
  **+0.1675** score gap and **+1.44 dB** at `segments=10` (score
  assertions against the real-data JSON skip in that mode; protocol
  assertions still run).
