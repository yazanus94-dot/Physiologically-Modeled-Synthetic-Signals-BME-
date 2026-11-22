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
  - achieved SNR improvement over the raw noisy signal (dB) — see
    `snr_improvement_db`,
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
  (`_zero_mean`) — documented in the reference solution.

## C. Starter baseline

A straightforward baseline (`starter_solution.py`):

1. Bandpass filter (0.5–40 Hz) on the noisy segment (zero-phase
   `sosfiltfilt`, order 3).
2. Return the filtered signal.

It scores modestly but is easy to understand; the reference solution
out-performs it by adding a **median baseline-removal** step and a
**50 Hz IIR notch** in front of the same band-pass:

| | reference | starter |
|---|---:|---:|
| `overall_score` | **0.8120** | 0.6443 |
| mean SNR improvement | **+3.16 dB** | +2.65 dB |
| signal-processing sub-score | **0.9192** | 0.6609 |
| edge robustness | **1.0** | 0.0 (raises on the 16-sample probe) |

Measured with `evaluator.evaluate(..., segments=10)` on the real data
(see `expected_results.json`); the offline synthetic fallback reproduces
the same ordering with a +0.158 score gap.  The starter's short-segment
failure is honest and reproducible: `sosfiltfilt` needs a longer pad
than the 16-sample edge probe provides.

## D. Reference solution

`reference_solution.py`: a complete, documented implementation mirroring
the primary denoising pipeline of Project 2 (`project-2-ecg-denoising`),
including:

- `bandpass_filter` (Butterworth 0.5–40 Hz, order 3, zero-phase via
  `sosfiltfilt`),
- `median_filter_baseline_removal` (median window 0.6 s),
- `notch_filter` (IIR notch, default 50 Hz),
- `fft_soft_threshold` (FFT-domain soft-threshold: out-of-band bins
  zeroed, in-band magnitudes shrunk by `0.08 * max|magnitude|` with the
  phase preserved),
- `combined_pipeline` (median → notch → bandpass).

**What the reference actually runs.**  `denoise(noisy, fs)` **is**
`combined_pipeline(noisy, fs, powerline_hz=50.0)` — median baseline
removal → 50 Hz notch → 0.5–40 Hz band-pass.  `fft_soft_threshold` is
provided as a documented, tested *alternative* stage (it is exercised by
the module's own helpers and is what Project 2's `fft` method uses), but
it is **not** part of the default pipeline; on this protocol it scores
below the combined chain (measured overall 0.7446 vs 0.8123 — the
combined chain reaches higher SNR gain *and* a higher overall score, so
it is the better submission; it is not what the reference submits).

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
- `performance_metrics` — diagnostics: `mean_snr_improvement_db`,
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

`tests/test_solutions.py` (30 tests) contains:

- **Success cases:** `reference_solution` on the evaluator → structure
  ok, finite outputs, `overall_score` within 0.05 of
  `expected_results.json` (protocol fields asserted in *both* modes, so
  CI without data still catches spec drift); `reference` beats
  `starter` by a measured margin (**> +0.05** overall, **> +0.10 dB**
  mean SNR improvement — measured +0.1677 / +0.62 dB on real data);
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
  reaches `rpeak_metrics`; F1 stays diagnostic-only; provenance block
  complete.
- **Edge cases:** very short segment (16 samples), zero-power input
  (graceful, no crash), short/zero inputs to the reference solution.

## G. Scoring rubric

| Criterion | Weight | Description |
|---|---:|---|
| Output structure | 15% | finite, 1-D, same shape, non-degenerate, `structure_ok=True` |
| Numerical validity | 10% | no NaN/Inf, no shape mismatch, sensible values |
| Signal-processing correctness | 25% | bandpass attenuates 50/60 Hz well; morphology preserved; energy concentrated in 0.5–40 Hz |
| Performance metrics | 25% | average SNR improvement, RMSE, NRMSE, PRD across cells — higher is better (scored by improving the mean) |
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
median → 50 Hz notch → 0.5–40 Hz band-pass):

- **`overall_score` = 0.8120** (structure 1.0, numerical validity 1.0,
  signal processing 0.9192, performance 0.3290, reproducibility 1.0,
  edge robustness 1.0).
- Mean SNR improvement **+3.16 dB** vs **+2.65 dB** for the starter;
  the reference is strictly better at *every* target SNR from −6 dB to
  +24 dB, and better at every SNR in mean SNR improvement too.
- Starter `overall_score` = 0.6443 (edge robustness 0.0: `sosfiltfilt`
  raises on the 16-sample edge probe) → margin **+0.1677**.
- Offline synthetic fallback reproduces the same ordering with a
  +0.158 score gap (score assertions against the real-data JSON skip in
  that mode; protocol assertions still run).
