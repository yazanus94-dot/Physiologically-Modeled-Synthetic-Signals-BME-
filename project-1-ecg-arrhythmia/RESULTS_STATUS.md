# Result provenance register — Project 1 (ECG arrhythmia)

## Scientific correction (2026-11-24)

`r_amp_rel` now reads the amplitude at the **annotation-centred R position**
(`round(0.25 * fs)` samples into the beat window) instead of the geometric centre of the
0.25 s pre / 0.45 s post window, which is ~100 ms after R (the ST/T region). This is a
scientific correction, not performance tuning. Related fixes in the same revision:
`record_info()` uses `sig_len` not `n_sig`; the real-data annotation helper filters to AAMI
beat symbols; the detector-driven false-positive count excludes boundary detections.

Regression tests: `tests/test_scientific_regressions.py` (4 tests, passing).

## Which committed results are current vs stale

| Result file | Data | Produced with corrected `r_amp_rel`? | Status |
|---|---|---|---|
| `outputs/synthetic/metrics.json` | synthetic generator | yes (regression suite re-run) | current for the synthetic pipeline |
| `outputs/detector_driven/metrics.json` | synthetic + detector | yes | current |
| `outputs/mitbih/classic/metrics.json` | real MITDB | **no** | **STALE — historical result** |
| `outputs/mitbih/leakage_audited/metrics.json` | real MITDB | **no** | **STALE — historical result** |

The two `mitbih/` metric files were generated before the feature correction. They are preserved
as historical evidence and must **not** be quoted as results of the current implementation.
No raw MITDB data is committed and none was downloaded for this audit, so they were not
regenerated.

## Reproduction (requires you to obtain the licensed MIT-BIH database)

```bash
# From the repository root; raw data must already be present locally.
make mitbih
# or individually:
cd project-1-ecg-arrhythmia/src
python run_pipeline.py --mode mitbih --protocol classic
python run_pipeline.py --mode mitbih --protocol leakage_audited
```

After regeneration, update the "Produced with corrected r_amp_rel?" column above and re-run
`python -m pytest -q project-1-ecg-arrhythmia/tests`.

Real MIT-BIH and BIDMC files are subject to PhysioNet terms; no dataset access, terms
acceptance or download was performed as part of this audit. These are research benchmarks,
not clinical validation.
