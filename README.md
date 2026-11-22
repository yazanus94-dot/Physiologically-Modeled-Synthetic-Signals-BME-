# Physiologically Modeled Synthetic Signals — Biomedical Engineering Projects

## 👋 Overview

A portfolio of three practice projects on physiological signal
processing with Python:

1. **ECG arrhythmia beat classification** — synthetic ECG generating
   Normal / PVC / Supraventricular beats, a from-scratch Pan-Tompkins-style
   R-peak detector, beat-level feature extraction, and Random Forest /
   Logistic Regression classifiers — plus the same pipeline run on the
   **real MIT-BIH Arrhythmia Database**.
2. **ECG denoising pipeline comparison** — objective scoring (SNR
   improvement, RMSE) of five denoising strategies on a synthetic ECG
   corrupted with known noise sources — plus the same methods scored on
   the **real PhysioNet Noise Stress Test Database**, where real recorded
   noise is mixed into real ECG at exact target SNRs.
3. **PPG-based heart rate variability (HRV) analysis** — synthetic PPG
   with respiratory-sinus-arrhythmia-driven HRV, systolic peak detection,
   and time- and frequency-domain HRV estimation — plus the same pipeline
   run on the **real BIDMC PPG and Respiration Dataset**, comparing
   ECG-derived HRV against PPG-derived PRV.

Every project keeps its synthetic and real results **separate and
labelled**. Synthetic numbers come from seeded generators; real numbers
come from downloaded PhysioNet recordings. They are never blended into a
single figure, and neither is presented as clinical performance.

**Downloaded PhysioNet data is never committed** — `data/` is git-ignored.
Each project documents how to re-download it.

## 📂 Repository structure

```
Physiologically-Modeled-Synthetic-Signals-BME-/
├── project-1-ecg-arrhythmia/      # ECG beat classification (RF + LR + PT detector)
├── project-2-ecg-denoising/       # denoising method comparison
├── project-3-ppg-hrv/             # PPG peak detection + HRV metrics
├── benchmark-tasks/               # PhysioNet-style benchmark package (TASK.md + evaluator)
├── .github/workflows/ci.yml       # CI: runs every project's + the benchmark test suites
├── requirements.txt               # top-level dependency set (installs all three)
├── versions.py                    # shared environment-version recorder (metrics provenance)
└── README.md
```

Each project folder is self-contained:

| Folder | Focus | Synthetic | Real (PhysioNet) |
|---|---|---|---|
| `project-1-ecg-arrhythmia/` | ECG beat classification | sum-of-Gaussians generator, Pan-Tompkins-style detector, RR/QRS features, RF + LR | MIT-BIH Arrhythmia Database (48 records) — DS1/DS2 split + leakage-audited split, R-peak scoring, **and a separate detector-driven evaluation** (features from the detector's own peaks, annotations used only as matched labels) |
| `project-2-ecg-denoising/` | denoising comparison | median baseline removal, Butterworth bandpass, notch, FFT soft-threshold, combined chain | MIT-BIH Noise Stress Test Database — real noise mixed into real ECG at exact SNRs |
| `project-3-ppg-hrv/` | PPG-based HRV | RSA-modulated generator, adaptive peak detection, SDNN/RMSSD/pNN50, Welch LF/HF | BIDMC PPG + Respiration Dataset — ECG-derived HRV vs PPG-derived PRV |
| `benchmark-tasks/` | self-contained benchmark package | — | `mitbih_nstdb_denoising_benchmark`: TASK.md (A–H spec) + starter/reference solutions + automated evaluator + `expected_results.json` (360 cells on real MIT-BIH 118/119 + NSTDB) |

## 📊 Headline results (all measured by the runs in this repository)

### Synthetic (seeded, regenerated live by each pipeline)

| Project | Metric | Result |
|---|---|---|
| 1 | R-peak detection (±50 ms) | 100.0% sensitivity / 100.0% precision |
| 1 | N/V/S beat classification | 1.00 accuracy (held-out split) |
| 2 | best denoising step | +16.34 dB SNR improvement (Butterworth 0.5–40 Hz) |
| 3 | peak detection (±150 ms) | 99.5% sensitivity / 100.0% precision |

### Real PhysioNet data

| Project | Dataset | Metric | Result |
|---|---|---|---|
| 1 | MIT-BIH (44 records) | R-peak detection (±50 ms, one-to-one) | sens 0.9434 / prec 0.9560 / F1 0.9497, mean timing error 3.79 ms |
| 1 | MIT-BIH, classic DS1/DS2 | beat classification (Logistic Regression) | balanced acc. 0.7063, macro-F1 0.5581 |
| 1 | MIT-BIH, leakage-audited | beat classification (Logistic Regression) | balanced acc. 0.7081, macro-F1 0.5689 |
| 1 | MIT-BIH, **detector-driven** (causal features) | beat classification (Random Forest) | balanced acc. 0.5014, macro-F1 0.4765 — *lower than annotation-centred on purpose* |
| 2 | NSTDB, 18 noise×SNR cells | bandpass SNR gain at −6 dB | **+17.00 dB** (combined +18.21 dB) — records **118, 119** |
| 2 | NSTDB, 18 noise×SNR cells | bandpass SNR gain at +24 dB | **−6.44 dB** (already-clean cost) |
| 2 | NSTDB R-peak (bandpass, predefined ±150 ms) | F1 | **0.606** (sens 0.9920 / prec 0.4390) |
| B | benchmark `mitbih_nstdb_denoising_benchmark` | reference `overall_score` on real data (360 cells) | **0.8120** vs starter **0.6443** (margin +0.1677) |
| 3 | BIDMC (3 records) | mean HR difference, HRV vs PRV | **−0.40 %**, beat agreement 97.3 % |
| 3 | BIDMC (3 records) | SDNN / RMSSD / LF-HF difference | 75.82 % / 84.58 % / 506.56 % |

**Read the caveats in each project README** — why these numbers must not
be read as clinical performance, and which failure modes are kept in the
results deliberately. In particular, Project 1's synthetic 1.00
classification accuracy versus its real balanced accuracy of ~0.71 is the
honest distance between a clean generator and patient-independent
classification on real ECG.

## 🚀 How to run

```bash
git clone https://github.com/yazanus94-dot/Physiologically-Modeled-Synthetic-Signals-BME-.git
cd Physiologically-Modeled-Synthetic-Signals-BME-

python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# --- synthetic pipelines ---
cd project-1-ecg-arrhythmia/src && python run_pipeline.py --mode synthetic && cd ../..
cd project-2-ecg-denoising/src  && python run_comparison.py            && cd ../..
cd project-3-ppg-hrv/src        && python run_pipeline.py              && cd ../..

# --- real PhysioNet pipelines (download data on first run) ---
cd project-1-ecg-arrhythmia/src && python run_pipeline.py --mode mitbih --protocol classic         && cd ../..
cd project-1-ecg-arrhythmia/src && python run_pipeline.py --mode mitbih --protocol leakage_audited && cd ../..
cd project-1-ecg-arrhythmia/src && python run_pipeline.py --mode rpeaks                            && cd ../..
#   detector-driven classification (annotations used only as matched labels):
cd project-1-ecg-arrhythmia/src && python run_detector_driven.py                                     && cd ../..
#   (optional) record-grouped hyperparameter tuning, on demand:
cd project-1-ecg-arrhythmia/src && python run_pipeline.py --mode mitbih --protocol classic --cv     && cd ../..
cd project-2-ecg-denoising/src  && python run_real_experiment.py --download                        && cd ../..
cd project-3-ppg-hrv/src        && python run_bidmc_experiment.py --download                       && cd ../..

# tests (275 total: 154 + 40 + 49 + 32 benchmark)
pytest project-1-ecg-arrhythmia/tests project-2-ecg-denoising/tests project-3-ppg-hrv/tests benchmark-tasks/mitbih_nstdb_denoising_benchmark/tests
```

Note: `--segments 10` is the default real-run configuration for Project 2
(10 independent noise segments per noise-type × SNR cell, 10 s segments,
records 118/119).  Record-grouped
cross-validation in Project 1 is available via `--cv`; it is off by
default because beat-level shuffled CV is deliberately unavailable.

Every pipeline writes its metrics (`metrics.json` / `*.csv`) and plots
(`*.png`) to its own `outputs/` folder. CI runs the same four test
suites (the three projects **and** `benchmark-tasks/`) plus Ruff over
`versions.py` and `benchmark-tasks/`, on Python 3.12.

## 📚 Datasets

Real data is downloaded from PhysioNet and **not** committed:

- **MIT-BIH Arrhythmia Database** — Moody, G., & Mark, R. (2005).
  *MIT-BIH Arrhythmia Database* (v1.0.0). PhysioNet.
  https://doi.org/10.13026/C2F305
- **MIT-BIH Noise Stress Test Database** — Moody, G. B., Muldrow, W. E.,
  & Mark, R. G. (1984). A noise stress test for arrhythmia detectors.
  *Computers in Cardiology*, 11, 381–384.
  https://doi.org/10.13026/C2HS3T
- **BIDMC PPG and Respiration Dataset** — Pimentel, M. A. F., et al.
  (2018). *BIDMC PPG and Respiration Dataset* (v1.0.0). PhysioNet.
  https://doi.org/10.13026/C2208R · original paper: Pimentel et al.
  (2016), *IEEE TBME*, 64(8), 1914–1923.

## 🎓 What it demonstrates

- Physiological signal modeling (sum-of-Gaussians ECG morphologies,
  asymmetric PPG pulses, RSA modulation of the tachogram)
- From-scratch classic algorithms (Pan-Tompkins QRS detection)
- Digital filtering designed per artifact type, not by trial and error
- Feature engineering driven by physiology (RR prematurity patterns,
  QRS width/energy for the wide PVC morphology)
- Reproducible real-data evaluation: patient-level train/test splits with
  a same-subject leakage audit (records 201/202), one-to-one peak matching
  at fixed tolerance, and explicitly reported artifact-rejection counts
- Honest evaluation: synthetic numbers labelled as synthetic, real numbers
  labelled as real, failure cases and negative results left in, and the
  documented limits of what each can and cannot establish

## 👨‍💻 About

**Yazan Aljabari** — Biomedical Engineering | Biomedical Signal
Processing | AI & Software Testing.
GitHub: [yazanus94-dot](https://github.com/yazanus94-dot)

---

**Disclaimer:** educational/demonstration code. Not a medical device;
nothing here is intended for clinical diagnosis or treatment.
