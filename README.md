# ECG Arrhythmia Beat Classification (Pan-Tompkins R-peak Detection + Random Forest)

Personal practice project applying a classic biomedical signal processing
pipeline to ECG beat classification: R-peak detection, beat-level feature
extraction, and a Normal / PVC / Supraventricular beat classifier.

## ⚠️ Data source — please read

This was built in an environment **without internet access** to PhysioNet,
so it uses a **physiologically-informed synthetic ECG generator**
(`src/generate_signal.py`), not real downloaded MIT-BIH Arrhythmia
Database records. The generator builds P-QRS-T morphology from a
sum-of-Gaussians model (the same style of approach used by tools like
ECGSYN), with realistic baseline wander, powerline interference, and
EMG-like noise added on top.

The analysis pipeline — R-peak detection, feature extraction,
classification — is written against the same interface `wfdb` uses for
real PhysioNet records, so swapping in real MIT-BIH data is a one-line
change. See `load_real_mitbih_record()` in `src/generate_signal.py` for
the exact `wfdb` calls, and `requirements.txt` for the one extra
dependency (`wfdb`) that step needs.

**This is disclosed here deliberately.** This is a learning/demonstration
project, not a submission to any PhysioNet/Computing in Cardiology
Challenge, and no claim is made otherwise.

## What it does

1. **Signal generation** (`generate_signal.py`) — 10 minutes of synthetic
   ECG at 250 Hz, with randomly interspersed Normal (N), PVC (V), and
   supraventricular/other (S) beats, each with distinct morphology.
2. **R-peak detection** (`detect_rpeaks.py`) — a Pan-Tompkins-style
   detector implemented from scratch with `scipy`: bandpass filter →
   derivative → squaring → moving-window integration → adaptive
   thresholding with a refractory period. Detector output is scored
   against the known ground-truth peaks (sensitivity/precision).
3. **Feature extraction** (`features.py`) — per-beat RR-interval features
   (pre/post RR, RR ratio — captures prematurity) and QRS morphology
   features (amplitude, width, energy — captures wide/bizarre PVC shape).
4. **Classification** (`run_pipeline.py`) — a Random Forest classifies
   each beat as N / V / S from the extracted features, evaluated on a
   held-out test split with a full classification report and confusion
   matrix.

## Results (this run, synthetic data)

- R-peak detector: **92.0% sensitivity, 100% precision** vs. ground truth
  (tolerance ±50 ms) — realistic for a from-scratch detector; it is not
  tuned to be artificially perfect.
- Beat classifier: near-perfect on held-out synthetic beats. This is
  expected and **not claimed as real-world performance** — the synthetic
  morphologies are more cleanly separable by construction than real,
  noisy clinical ECG, where normal/ectopic beats and inter-patient
  variability make this a genuinely harder problem. On real MIT-BIH data,
  published inter-patient beat classifiers typically score well below
  this.

See `outputs/metrics.json` for the full numeric report and
`outputs/*.png` for the plots (ECG strip with detected peaks, confusion
matrix, feature importance).

## Run it

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py
```

Output (metrics + plots) is written to `outputs/`.

## Project structure

```
proj1_ecg_arrhythmia/
├── README.md
├── requirements.txt
├── src/
│   ├── generate_signal.py   # synthetic ECG generator + wfdb swap-in notes
│   ├── detect_rpeaks.py     # from-scratch Pan-Tompkins-style R-peak detector
│   ├── features.py          # beat-level feature extraction
│   └── run_pipeline.py      # end-to-end: generate -> detect -> extract -> classify
└── outputs/                 # generated on run: metrics.json + plots
```
