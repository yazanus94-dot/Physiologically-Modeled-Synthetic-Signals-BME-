# PhysioNet-style Benchmark Task — MIT-BIH + NSTDB ECG Denoising

Independent, self-contained benchmark package in the spirit of
PhysioNet-style scientific evaluation.  **Not an official PhysioNet
Challenge submission.**

| File | Purpose |
|---|---|
| [`TASK.md`](TASK.md) | A–H package specification (problem, dataset/protocol, baseline, reference, evaluator, tests, rubric, expected results) |
| [`starter_solution.py`](starter_solution.py) | Straightforward band-pass baseline |
| [`reference_solution.py`](reference_solution.py) | Complete, documented reference implementation |
| [`evaluator.py`](evaluator.py) | Automated evaluator (structure / validity / signal-processing / performance / reproducibility / edge cases) |
| [`expected_results.json`](expected_results.json) | Reference results on real data + reproducible command |
| [`tests/test_solutions.py`](tests/test_solutions.py) | Success + failure-case tests |

## Run it

```bash
# from this directory
python -m pytest tests                     # full test suite (uses real data when present)
python evaluator.py --solution reference_solution
```

The evaluator loads real PhysioNet data via the Project 2 loaders
(`project-2-ecg-denoising/src/load_real_ecg.py` /
`load_real_noise.py`) when they are importable and the data is present
(`<repo>/data/physionet/`).  Otherwise it falls back to a **synthetic**
signal set so the evaluator is runnable offline — the tests handle both
modes (`expected_results.json` was generated with the real data).

## Scoring

See `TASK.md` (section G) and `evaluator.SCORING_RUBRIC`:

| Criterion | Weight |
|---|---:|
| Output structure (finite, 1-D, same shape) | 15 % |
| Numerical validity | 10 % |
| Signal-processing correctness (50/60 Hz suppression) | 25 % |
| Performance (mean error-based SNR improvement across cells) | 25 % |
| Reproducibility (deterministic across runs) | 15 % |
| Edge-case robustness (short/zero segments) | 10 % |

`overall_score` is the weighted average in `[0, 1]`.

## Measured reference-vs-starter (real data, 360 cells)

| | reference | starter |
|---|---:|---:|
| `overall_score` | **0.8366** | 0.6469 |
| mean error-based SNR improvement | **+3.89 dB** | +2.99 dB |
| edge robustness | **1.0** | 0.0 |
