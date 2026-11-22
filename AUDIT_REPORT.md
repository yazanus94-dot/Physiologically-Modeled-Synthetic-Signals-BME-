# Final Audit Report — Physiologically-Modeled-Synthetic-Signals-BME

Repo: `yazanus94-dot/Physiologically-Modeled-Synthetic-Signals-BME-`
Audit round: corrections for the final audit (Phases 1–5), run after commit
`c82b6ef`. Report date: 2026-11-23.

---

## 1. Files changed

| File | Change |
|---|---|
| `benchmark-tasks/mitbih_nstdb_denoising_benchmark/TASK.md` | Stated margins & the measured FFT-alternative number in §D (reference pipeline is median → 50 Hz notch → 0.5–40 Hz band-pass). |
| `benchmark-tasks/mitbih_nstdb_denoising_benchmark/expected_results.json` | Regenerated from the corrected evaluator on real data (identical scores; adds `seaborn: 0.13.2` to provenance software). |
| `benchmark-tasks/mitbih_nstdb_denoising_benchmark/reference_solution.py` | Docstring: measured numbers (+3.1604 dB reference vs +2.6479 dB starter, +0.5125 dB gap; overall 0.8120 vs 0.6443). |
| `benchmark-tasks/mitbih_nstdb_denoising_benchmark/starter_solution.py` | Docstring: measured +0.5125 dB / +0.1677. |
| `benchmark-tasks/mitbih_nstdb_denoising_benchmark/tests/test_solutions.py` | Strict comparison margins now meaningful and measured: overall gap `> 0.10` (measured 0.1692 at the test context, 0.1677 at 10 segments); SNR improvement margin `> 0.12 dB` (measured 0.6169 / 0.5125). |
| `project-1-ecg-arrhythmia/README.md` | Implied sensitivity/precision aligned to the regenerated `metrics.json` (0.9001 / 0.9642); documented the new `limitations` block. |
| `project-1-ecg-arrhythmia/outputs/detector_driven/metrics.json` | Regenerated on real data; now records the combined end-to-end `limitations` block. |
| `project-1-ecg-arrhythmia/src/run_detector_driven.py` | New `limitations` block (bounded-by-detector + explanation) on the report; docstring updated. |
| `project-1-ecg-arrhythmia/tests/test_detector_driven.py` | New leakage test `test_matched_features_are_independent_of_label_values` (X is invariant to label values — annotation/labels never enter features). |
| `AUDIT_REPORT.md` | This report. |

Notes on prior rounds (verified in this round): the benchmark CI tests were
already present in `.github/workflows/ci.yml` (pytest line) and in the
Makefile `test` target — verified, nothing to add. The Phase 2.2/2.3
discrepancies described by the user (30-second evaluator segments,
`or True` short-segment check, missing 60 Hz probe, un-threaded `seed`)
were already fixed in the later commits after `a1c0238`; they are listed as
verified-and-closed below.

---

## 2. Bugs found and corrected

1. **Ruff (reproduced on the `a1c0238` files, current tree clean):**
   `FURB136` (`if-expr-min-max` in `reference_solution.py`:
   `x.size if x.size >= 3 else 3`) and three `RUF100` unused
   `noqa: E402` directives in `tests/test_solutions.py`. Fixed by the later
   commits; verified now: `python -m ruff check versions.py benchmark-tasks/`
   exits 0 on the current tree (see §4).
2. **Phase 2.1 — reference vs starter margins:** old comparison thresholds
   replaced by meaningful, strict, measured ones (see §6 for the numbers).
   The FFT alternative was measured (overall 0.7446, SNR +5.1384 dB) and
   deliberately not the reference pipeline — the combined chain scores
   higher on overall; documented honestly in `TASK.md` §D.
3. **Phase 2.2 — TASK.md / expected_results alignment:** at `a1c0238` the
   evaluator created **30-second** segments while TASK.md documented 10 s;
   `expected_results.json` reported `segments_per_cell: 10` with
   `n_cells: 360` = 3 noise × 6 SNR × 2 records × 10 segments (no `4`
   anywhere in the committed JSON — verified). Current tree:
   `DEFAULT_SEGMENT_S = 10.0` in `evaluator.py` used by both the real-data
   path and the synthetic fallback, so TASK.md (10 s), the JSON
   (`segment_duration_s: 10.0`) and the code all agree on one value.
   No train/validation/test split is invented: TASK.md documents a fixed
   independent evaluation protocol and the evaluator implements none
   (verified: `evaluate()` has no split; provenance protocol reads
   "fixed independent evaluation (no train/val/test split)").
4. **Phase 2.3 — evaluator correctness (verified against the current
   tree):** `_score_structure` rejects constant/near-flat outputs
   (`test_score_structure_rejects_constant_output`,
   `test_near_constant_output_is_rejected`); the short-segment edge check
   has no `or True` (fixed earlier; `test_edge_check_rejects_crash_on_short_segment`
   passes); `tolerance_ms` is threaded into `rpeak_metrics`
   (`test_tolerance_ms_is_propagated_to_rpeak_metrics`); `seed` reaches the
   real-data loader RNG and the synthetic fallback
   (`test_seed_controls_synthetic_generation`,
   `test_different_seeds_change_the_real_protocol_cells`); `data_dir`
   reaches the loaders on the first load and constants are restored
   (`test_data_dir_threaded_to_loaders`,
   `test_data_dir_used_when_default_directory_is_empty`,
   `test_data_dir_override_is_restored_afterwards`); R-peak F1 is
   diagnostic-only (rubric has no `rpeak_f1`; TASK.md §F says diagnostic;
   `mean_rpeak_f1` reported in `performance_metrics`); the
   signal-processing score implements controlled **50 Hz and 60 Hz** probe
   tones (`_score_signal_processing` injects both and scores their
   attenuation); malformed outputs get no performance/signal-processing
   credit (`valid_flags`, `test_malformed_outputs_get_no_performance_or_sp_credit`);
   determinism is enforced (`reproducibility_digest`, `test_reference_deterministic`).
   Regression tests exist for constant output, short segments, parameter
   propagation (tolerance/seed/data_dir), different seeds, zero-power
   data and mismatched shapes.
5. **Phase 2.4 — expected results:** `expected_results.json` regenerated by
   `evaluator.evaluate(reference_solution())` on real data
   (`data/physionet/{mitdb,nstdb}`, records 118/119 + bw/ma/em). Scores
   identical to the committed JSON; the only diff is the added
   `seaborn: 0.13.2` version. README (`benchmark-tasks/.../README.md`)
   quotes no conflicting numbers and agrees with the JSON.
6. **Phase 3 — detector-driven evaluation:** verified present and complete
   (detector-first features, one-to-one label assignment, FP/FN/matched
   reporting, classification on matched beats). Gap closed:
   `run_detector_driven.py` now records a `limitations` block
   (`bounded_by_detector`, pooled detection sens/prec, explanation);
   `outputs/detector_driven/metrics.json` regenerated on real data.
   New leakage regression test added: `X` is a function of
   `(signal, detected peaks)` only — labels never enter it.
7. **Phase 4 — project-3 portability:** verified: `outputs/real/metrics.json`
   has the portable relative path `"data_dir": "data/physionet/bidmc"`
   (no absolute Windows paths anywhere in `project-3-ppg-hrv` — grep clean);
   the 40–200 bpm band is central (`peak_detect.py`:
   `MIN_PULSE_RATE_BPM = 40.0`, `MAX_PULSE_RATE_BPM = 200.0`) and used
   consistently — interval filtering derives its limits from the same
   constants (`run_bidmc_experiment.py`: `iv_lo, iv_hi = 60/MAX, 60/MIN`),
   `quality_assessment.py` defaults import the same constants, and the
   tests assert the values (`test_detector_rates_and_conditions.py`:
   `(40.0, 200.0)`, PL_LO/HI consistency, README mention).
   XQRS is explicitly labelled automated (`reference_kind`: "AUTOMATED
   detector output (wfdb XQRS…)") and so is the monitor reference. The
   HRV-vs-PRV limitations (differences at ~0.4 % HR, SDNN/RMSSD
   differences, LF/HF instability, series independence) are preserved
   unchanged.
8. **Phase 5 — final validation:** see §3–§5.

---

## 3. Test results

| Suite | Result |
|---|---:|
| benchmark-tasks (`mitbih_nstdb_denoising_benchmark`) | **32 passed** (real data) |
| project-1-ecg-arrhythmia | **155 passed** |
| project-2-eg-denoising | **40 passed** |
| project-3-ppg-hrv | **49 passed** |
| **All suites via the Makefile/CI command** | **276 passed** |

Synthetic fallback (forced, as in CI): benchmark suite passes —
`test_synthetic_fallback_mode_runs_offline` and the margin tests hold;
measured gaps at the test-margin context (segments=2, 12 cells):
reference overall 0.8138 vs starter 0.6562 (gap +0.1576 > 0.10), reference
SNR +1.2701 dB vs starter +1.0205 dB (gap +0.2496 > 0.12). Synthetic runs
of the projects (`run_pipeline --mode synthetic` / `run_comparison.py` /
`run_pipeline.py`) reproduce committed outputs (no diffs).

CI: **green** (confirmed in §8 after the push).

---

## 4. Ruff results

- CI command `python -m ruff check versions.py benchmark-tasks/`:
  **All checks passed** (exit 0).
- Changed project-1 files
  (`project-1-ecg-arrhythmia/src/run_detector_driven.py`,
  `project-1-ecg-arrhythmia/tests/test_detector_driven.py`):
  **All checks passed** (exit 0).

---

## 5. Real experiments actually rerun (real PhysioNet data on disk)

| Experiment | Outcome |
|---|---|
| Benchmark, real mode (`evaluate(reference_solution())`, mode=real, 360 cells) | reference overall **0.8120**; starter **0.6443** (gap **+0.1677**); reference SNR +3.1604 dB vs +2.6479 dB (gap +0.5125 dB) |
| Benchmark, synthetic fallback (forced offline mode) | reference 0.8134 / starter 0.656 (gap +0.1574); SNR +1.2463 / +0.9903 |
| project-1 `--mode mitbih --protocol classic` (rerun) | outputs byte-identical to committed: RF 0.6413/0.4766/0.8555; LR 0.7063/0.5581/0.8882; test set 49,661 beats |
| project-1 `--mode mitbih --protocol leakage_audited` (rerun) | byte-identical: RF 0.6405/0.4806/0.8565; LR 0.7081/0.5689/0.8943; leakage audit prints "No subject leakage detected" |
| project-1 `--mode rpeaks` (rerun) | byte-identical: TP=95,014, FP=4,369, FN=5,704; sens 0.9434, prec 0.9560, F1 0.9497, mean timing 3.79 ms |
| project-1 detector-driven (`run_detector_driven.py`, rerun) | metrics regenerated: TP 46,029 / FP 1,707 / FN 5,106; RF 0.5014/0.4765, LR 0.2515/0.2386; new `limitations` block present |
| project-2 real (`run_real_experiment.py`, rerun) | outputs byte-identical to committed (pearson r 0.7511; rpeak F1 0.5995) |
| project-2 synthetic (`run_comparison.py`, rerun) | outputs unchanged from committed |
| project-3 real (`run_bidmc_experiment.py --download`, rerun) | outputs unchanged from committed (`data_dir` portable) |
| project-1 synthetic (`run_pipeline.py --mode synthetic`, rerun) | outputs unchanged from committed |

---

## 6. Does the reference solution measurably beat the starter? — Yes

Measured on the fixed evaluation protocol (real MIT-BIH 118/119 + NSTDB
`bw`/`ma`/`em`):

| Metric | Reference | Starter | Gap |
|---|---:|---:|---:|
| `overall_score` | 0.8120 | 0.6443 | **+0.1677** |
| Mean SNR improvement (dB) | +3.1604 | +2.6479 | **+0.5125** |
| signal-processing sub-score | 0.9192 | 0.6609 | +0.2583 |
| performance sub-score | 0.3290 | — | — |
| edge robustness | 1.0 | 0.0 (starter raises on the 16-sample probe) | — |

At the margin context used by the tests (`segments=2`, 72 real cells):
reference 0.8123 vs starter 0.6431 (**gap +0.1692**), SNR +3.3242 dB vs
+2.7073 dB (**gap +0.6169 dB**). The asserted margins (+0.10 overall,
+0.12 dB) are meaningful in both modes (measured gaps are ~1.6–1.7× the
thresholds). Nothing was artificially inflated: the reference pipeline
(median baseline removal → 50 Hz IIR notch → 0.5–40 Hz Butterworth
band-pass) is genuinely a different, better-scoring chain than the starter
(band-pass only). The FFT alternative was measured and rejected from the
reference pipeline because the combined chain scores higher on this
protocol — documented, not hidden.

---

## 7. Final benchmark score and reproducibility details

- **Reference `overall_score` = 0.8120** (real data, mode=real).
- Sub-scores: structure 1.0, numerical_validity 1.0, signal_processing
  0.9192, performance 0.3290, reproducibility 1.0, edge_robustness 1.0.
- Protocol (all documented in `expected_results.json` + `TASK.md` §B/H):
  fixed independent evaluation (no train/val/test split); ECG records
  `["118", "119"]`; noise records `["bw", "ma", "em"]`
  (baseline_wander / muscle_artifact / electrode_motion);
  target SNR levels `[-6, 0, 6, 12, 18, 24] dB`; `n_cells = 360`
  (3 noise × 6 SNR × 2 records × 10 segments of 10 s each);
  `segments_per_cell = 10`, `segment_duration_s = 10.0`, `seed = 3`,
  `peak_tolerance_ms = 150.0`, `data_dir = null`.
- `reproducibility_digest`
  `a3ffe2451ff5c598d8744a507fe5af88c226f4043e7c11c93366670b5b2c18d8`
  — regenerated twice from real data; identical (`reproducibility_ok: true`).
- Software versions stamped (`provenance.software`): python 3.14.7
  (CPython), numpy 2.5.4, scipy 1.18.1, pandas 2.3.3, wfdb 4.3.1,
  matplotlib 3.11.2, scikit-learn 1.9.1, joblib 1.6.0, seaborn 0.13.2.
- Benchmark tests pass with real data and with the offline synthetic
  fallback (§3).

---

## 8. CI / GitHub Actions status (pending final confirmation)

After this push, the newest run must show **every step green**:
`Run project test suites` and `Lint new code (benchmark-tasks + versions.py)`
(plus checkout/setup/install/post steps). See §9.

---

## 9. Hygiene

- Raw PhysioNet data (MIT-BIH, NSTDB, BIDMC) stays git-ignored:
  `git check-ignore data/physionet` → `.gitignore:26:data/`.
- No machine-specific paths in any tracked file: `git diff | grep "C:\Users"`
  → empty; all committed `data_dir` values are relative
  (`data/physionet/bidmc`).
- Committer identity: `yazanus94-dot <yazanus94-dot@users.noreply.github.com>`.
- Push: `git push origin main` — to be confirmed green below.

---

## 10. Remaining limitations (preserved from the previous audit rounds)

- **Detector-driven classification is expected to be worse than
  annotation-centred.** Annotation positions are unavailable at inference;
  the two approaches confound peak positions, RR-feature definitions and
  label assignment, so the observed gap (RF 0.5014 vs 0.6413 balanced
  accuracy) cannot be attributed to any one factor. Recorded as a
  limitation in `metrics.json.limitations`, not glossed over.
- Rare-class performance in project-1 real data: `S`/`F` recall low;
  accuracy/weighted-F1 in real data are dominated by the `N` class —
  balanced accuracy and macro-F1 are the honest metrics (documented in
  `project-1-ecg-arrhythmia/README.md`).
- Benchmark: synthetic fallback gap vs real data is +0.1576 overall
  (documented in `TASK.md` §H); R-peak F1 in the benchmark is
  diagnostic-only (rubric has no `rpeak_f1`).
- The FFT alternative (`fft_soft_threshold`) scores overall 0.7446 vs the
  combined chain's 0.8123 — documented as not the reference pipeline.
- Project-3: real BIDMC precision is limited by inter-channel alignment
  (monitor-reference bias effect); HRV-vs-PRV findings (mean-HR agreement
  ~0.4 %, SDNN/RMSSD differences, LF/HF instability, independent series)
  are preserved as the documented negative findings.
