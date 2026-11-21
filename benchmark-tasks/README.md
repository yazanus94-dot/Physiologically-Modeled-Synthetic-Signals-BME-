# Benchmark Tasks (PhysioNet-style)

Independent portfolio benchmarks inspired by PhysioNet-style scientific
evaluation.  Each package is **self-contained** and uses the real
PhysioNet data where it is available.

## Packages

| Package | Target | Benchmark source |
|---|---|---|
| [`mitbih_nstdb_denoising_benchmark`](mitbih_nstdb_denoising_benchmark/README.md) | ECG denoising | Project 2 (MIT-BIH + NSTDB) |

**Not official PhysioNet Challenges.** The benchmarks are designed to look
and evaluate like PhysioNet-style scientific tasks; the intended use is
solving them as code-quality proofs of engineering rigor — not as
competitions.

## Adding a new benchmark

A benchmark package contains (see
`mitbih_nstdb_denoising_benchmark/` for a complete example):

1. `TASK.md` — A–H specification (problem statement, dataset/protocol,
   starter baseline, reference solution, automated evaluator, tests/failure
   cases, scoring rubric, expected results).
2. `solution*.py` — baseline and reference solutions.
3. `evaluator.py` — automated evaluation (structure / validity /
   signal-processing / performance / reproducibility / edge cases).
4. `tests/test_*.py` — success and failure-case tests.
5. `expected_results.json` — reference outputs + reproducible command.
6. `README.md` — package overview.

## Status (as of 2026-11-21)

- [x] `mitbih_nstdb_denoising_benchmark` — implemented, evaluated against
      real PhysioNet data, committed.
