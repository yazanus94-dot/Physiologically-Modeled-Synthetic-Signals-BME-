# Physiologically Modeled Synthetic Signals — Biomedical Engineering Projects

## 👋 Overview

A portfolio of three practice projects on physiological signal
processing with Python:

1. **ECG arrhythmia beat classification** — synthetic ECG generating
   Normal / PVC / Supraventricular beats, a from-scratch Pan-Tompkins-style
   R-peak detector, beat-level feature extraction, and a Random Forest
   classifier.
2. **ECG denoising pipeline comparison** — objective scoring (SNR
   improvement, RMSE) of five denoising strategies on an ECG corrupted
   with known noise sources.
3. **PPG-based heart rate variability (HRV) analysis** — synthetic PPG
   with respiratory-sinus-arrhythmia-driven HRV, systolic peak detection,
   and time- and frequency-domain HRV estimation.

All data is **synthetic and seeded** — the signals were generated, not
recorded, which is disclosed in every project README along with the
reason (no internet access to PhysioNet) and the exact swap-in path for
real recordings. The honest consequence: every reported number describes
performance on this synthetic dataset, not real clinical performance.

## 📂 Repository structure

```
Physiologically-Modeled-Synthetic-Signals-BME-/
├── project-1-ecg-arrhythmia/      # ECG beat classification (RF + PT detector)
├── project-2-ecg-denoising/       # denoising method comparison
├── project-3-ppg-hrv/             # PPG peak detection + HRV metrics
├── .github/workflows/ci.yml       # CI: runs every project's test suite
├── requirements.txt               # top-level dependency set
└── README.md
```

Each project folder is self-contained:

| Folder | Focus | Key techniques |
|---|---|---|
| `project-1-ecg-arrhythmia/` | ECG beat classification | sum-of-Gaussians generator, Pan-Tompkins-style detector, RR/QRS features, Random Forest |
| `project-2-ecg-denoising/` | denoising comparison | median baseline removal, Butterworth bandpass, notch, FFT soft-threshold, combined chain |
| `project-3-ppg-hrv/` | PPG-based HRV | RSA-modulated generator, adaptive peak detection, SDNN/RMSSD/pNN50, Welch LF/HF |

## 📊 Headline results (synthetic data, regenerated live by each pipeline)

| Project | Metric | Result |
|---|---|---|
| 1 | R-peak detection (±50 ms) | 100.0% sensitivity / 100.0% precision |
| 1 | N/V/S beat classification | 1.00 accuracy (held-out split) |
| 2 | best denoising step | +7.2 dB SNR improvement (Butterworth 0.5–40 Hz) |
| 3 | peak detection (±150 ms) | 99.5% sensitivity / 100.0% precision |
| 3 | HRV agreement | SDNN/RMSSD within ~5% of reference |

**Read the caveats in each project README** — why these numbers must not
be read as clinical performance, and which failure modes were kept in
the results deliberately.

## 🚀 How to run

```bash
git clone https://github.com/yazanus94-dot/Physiologically-Modeled-Synthetic-Signals-BME-.git
cd Physiologically-Modeled-Synthetic-Signals-BME-

python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# Project 1
cd project-1-ecg-arrhythmia/src && python run_pipeline.py && cd ../..

# Project 2
cd project-2-ecg-denoising/src && python run_comparison.py && cd ../..

# Project 3
cd project-3-ppg-hrv/src && python run_pipeline.py && cd ..

# tests
pytest project-1-ecg-arrhythmia/tests project-2-ecg-denoising/tests project-3-ppg-hrv/tests
```

Every pipeline writes its metrics (`metrics.json`) and plots (`*.png`)
to its own `outputs/` folder.

## 🎓 What it demonstrates

- Physiological signal modeling (sum-of-Gaussians ECG morphologies,
  asymmetric PPG pulses, RSA modulation of the tachogram)
- From-scratch classic algorithms (Pan-Tompkins QRS detection)
- Digital filtering designed per artifact type, not by trial and error
- Feature engineering driven by physiology (RR prematurity patterns,
  QRS width/energy for the wide PVC morphology)
- Honest evaluation on synthetic data: known ground truth enables
  objective detection/classification scoring and the documented limits
  of what that can and cannot establish

## 👨‍💻 About

**Yazan Aljabari** — Biomedical Engineering | Biomedical Signal
Processing | AI & Software Testing.
GitHub: [yazanus94-dot](https://github.com/yazanus94-dot)

---

**Disclaimer:** educational/demonstration code. Not a medical device;
nothing here is intended for clinical diagnosis or treatment.
