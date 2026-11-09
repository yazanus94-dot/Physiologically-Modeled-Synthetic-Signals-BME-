# ECG Denoising Pipeline Comparison

Personal practice project comparing four classic denoising approaches
(and a realistic combined clinical pipeline) on the same noisy ECG
signal — where the clean ground truth is known, so SNR improvement and
RMSE are objective scores rather than visual impressions.

## ⚠️ Data source — please read

Built without internet access to PhysioNet: the ECG is
**synthetic** (`src/generate_noisy_signal.py`), a sum-of-Gaussians
clean generator with three independently-injected artifact sources on
top — baseline wander, 50 Hz powerline interference, and EMG-like
broadband noise. That clean reference is exactly what enables the
objective scoring.

**This is a learning/demonstration project**, not a clinical tool, and
no clinical claim is made or implied.

## What it does

1. **Corruption** (`generate_noisy_signal.py`) — 30 s of clean synthetic
   ECG at 250 Hz, corrupted with:
   - baseline wander (respiration/movement drift),
   - 50 Hz powerline pickup,
   - EMG-like broadband noise.
2. **Denoising methods** (`denoise_methods.py`):
   1. Median-filter baseline removal (classic clinical baseline-wander
      correction)
   2. Butterworth bandpass filtering (standard 0.5–40 Hz ECG passband)
   3. IIR notch filter targeted at the powerline frequency
   4. FFT-domain soft-thresholding denoiser (a frequency-domain analogue
      of wavelet soft-thresholding, without needing a wavelet library)
   5. A combined pipeline (1 → 3 → 2), the shape of a real clinical
      front-end chain
3. **Evaluation + plots** (`run_comparison.py`) — SNR and RMSE against
   the clean reference for every method, plus a comparison strip plot
   and an SNR-improvement bar chart.

## Run it

```bash
pip install -r requirements.txt
cd src
python run_comparison.py
```

Output (metrics + plots) is written to `outputs/`.

## Results (this run)

Numbers are regenerated live by the pipeline — `outputs/metrics.json`
is the authoritative record of every run. Current build (30 s at 250 Hz,
seed 3, baseline wander + 50 Hz powerline + EMG on top of a clean ECG at
raw SNR −0.85 dB):

| Method | SNR improvement | RMSE |
|---|---:|---:|
| Butterworth bandpass (0.5–40 Hz) | **+7.17 dB** | 0.0940 |
| Combined pipeline (median → notch → bandpass) | +6.99 dB | 0.0959 |
| Median-filter baseline removal | +6.82 dB | 0.0978 |
| FFT soft-thresholding | +5.20 dB | 0.1178 |
| 50 Hz notch alone | +0.33 dB | 0.2065 |

Straightforward reading: the bandpass gives the biggest single-step
improvement because it removes all three artifact classes at once; the
combined clinical-style pipeline is close behind while retaining each
step's specific role; the notch alone is nearly neutral — expected,
since powerline content is a small fraction of the total corruption on
its own.

See `outputs/metrics.json` for exact numbers, and
`outputs/denoising_comparison.png` / `outputs/snr_improvement.png` for
the plots.

## Project structure

```
project-2-ecg-denoising/
├── README.md
├── requirements.txt
├── src/
│   ├── generate_noisy_signal.py   # clean synthetic ECG + 3 noise sources
│   ├── denoise_methods.py         # the four denoising approaches + combined pipeline
│   └── run_comparison.py          # end-to-end: corrupt -> denoise -> score -> plot
├── tests/
│   └── test_pipeline.py           # fast unit tests of generator + methods
└── outputs/                       # generated on run: metrics.json + plots
```
