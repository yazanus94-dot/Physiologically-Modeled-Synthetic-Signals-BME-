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
- **Train / validation / test separation:** the task ships a fixed
  **patient-level split**: clean ECG records 118/119 → the evaluator runs
  over the train/val/test cell structure defined in
  `evaluator.py` (`TRAIN_ECG_RECORDS`, `TEST_ECG_RECORDS`,
  `TRAIN_NOISE_RECORDS`, `TEST_NOISE_RECORDS`). The mixed segments are
  generated deterministically from seeded RNGs.
- **One-to-one peak matching** at ±150 ms (predefined, never tuned) is
  used for the R-peak component of the score.
- **DC-offset convention:** waveform-error metrics (`rmse`, `nrmse`, `prd`,
  `snr_db`) are computed after removing the mean from both signals
  (`_zero_mean`) — documented in the reference solution.

## C. Starter baseline

A straightforward baseline (`starter_solution.py`):

1. Bandpass filter (0.5–40 Hz) on the noisy segment (zero-phase
   `sosfiltfilt`, order 3).
2. Return the filtered signal.

It scores modestly but is easy to understand; the reference solution
out-performs it by adding a median-baseline-removal step and an FFT
soft-threshold denoiser.

## D. Reference solution

`reference_solution.py`: a complete, documented implementation mirroring
the primary denoising pipeline of Project 2 (`project-2-ecg-denoising`),
including:

- `bandpass_filter` (Butterworth 0.5–40 Hz, order 3, zero-phase via
  `sosfiltfilt`),
- `median_filter_baseline_removal` (median window 0.6 s),
- `notch_filter` (IIR notch, default 50 Hz),
- `fft_soft_threshold_denoise` (FFT-domain soft-threshold at 0.08 of
  band power),
- `combined_pipeline` (median → notch → bandpass).

Each is documented with the parameter choices and the scientific rationale.
The reference implementation returns a `DenoisingResult` dataclass
(`signal`, `method`) and is reproducible (seeded).

## E. Automated evaluator

`evaluator.py` exposes:

```python
def evaluate(solution, records=(118, 119), snrs=(-6.0, 0.0, 6.0, 12.0, 18.0, 24.0),
             segments=10, seed=3, data_dir=DEFAULT_DATA_DIR,
             tolerance_ms=150.0) -> dict
```

Returns a dict with:

- `overall_score` — the weighted score (see `SCORING_RUBRIC`),
- `structure_ok` — boolean: output is finite, 1-D, same shape as input,
- `per_cell` — per (noise_category, snr) row:
  `{method, snr_improvement_db, rmse, nrmse, prd, pearson,
    rpeak_sensitivity, rpeak_precision, rpeak_f1, achieved_snr_db, ...}`,
- `reproducibility_digest` — hash of rounded output values (determinism),
- `summary` — means across cells.

The evaluator **loads the PhysioNet data lazily** via
`project-2-ecg-denoising/src/load_real_ecg.py` and
`load_real_noise.py` (or, if those are unavailable, falls back to bundled
synthetic reference data so the evaluator is runnable offline — the task
tests do this too).

## F. Tests and failure cases

`tests/test_solutions.py` contains:

- **Success cases:** `reference_solution` on the evaluator → structure_ok,
  finite outputs, overall_score within tolerance of `expected_score`;
  `starter_solution` scores lower than `reference_solution`.
- **Failure cases:**
  - Solution that returns the noisy signal unchanged → expect negative/zero
    SNR improvement.
  - Solution that returns a constant signal → structure_ok False
    (zero range), expect failure.
  - Solution that returns NaN/Inf → structure_ok False.
  - Solution with wrong shape → structure_ok False.
  - Reproducibility: running twice gives identical `reproducibility_digest`.
- **Edge cases:** very short segment (100 samples), high SNR (+24 dB),
  zero-power input (expect graceful failure, not crash).

## G. Scoring rubric

| Criterion | Weight | Description |
|---|---:|---|
| Output structure | 15% | finite, 1-D, same shape, `structure_ok=True` |
| Numerical validity | 10% | no NaN/Inf, no shape mismatch, sensible values |
| Signal-processing correctness | 25% | bandpass attenuates 50/60 Hz well; morphology preserved; energy concentrated in 0.5–40 Hz |
| Performance metrics | 25% | average SNR improvement, RMSE, NRMSE, PRD across cells — higher is better (scored by improving the mean) |
| Reproducibility | 15% | deterministic output digest for fixed seed |
| Edge-case robustness | 10% | graceful handling of short segments, zero-power inputs, constant signals |

`overall_score` = weighted average of the sub-scores in [0, 1].

## H. Expected results

`expected_results.json` — reference outputs regenerated from the repo's
Project 2 real-data experiment (records 118/119, 10 segments/cell,
tolerance 150 ms). Reproduce with:

```bash
cd project-2-ecg-denoising/src && python run_real_experiment.py --download
```

Key reference values (means over cells, bandpass method):

- Bandpass SNR improvement at −6 dB ≈ **+17.00 dB**; at +24 dB ≈ **−6.44 dB**
  (a correct high-pass removes signal along with no noise at high SNR —
  the evaluator scores this honestly).
- Bandpass R-peak: sens ≈ 0.992, prec ≈ 0.439, F1 ≈ 0.606 (150 ms tol).
- Reproducible digest recorded per method.
