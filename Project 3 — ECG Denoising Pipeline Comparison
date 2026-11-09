# PPG-Based Heart Rate Variability (HRV) Estimation

Personal practice project: systolic peak detection from a PPG
(photoplethysmogram) waveform, and time-domain + frequency-domain HRV
analysis from the resulting pulse-to-pulse interval series — following
the Task Force (ESC/NASPE) HRV metric conventions (SDNN, RMSSD, pNN50,
LF/HF ratio).

## ⚠️ Data source — please read

Built without internet access to PhysioNet, so this uses a
**physiologically-motivated synthetic PPG generator**
(`src/generate_ppg.py`) — an asymmetric double-Gaussian pulse model
(systolic peak + dicrotic notch) with realistic **respiratory sinus
arrhythmia** modulation (the dominant real physiological driver of
short-term HRV: heart rate oscillates with the breathing cycle) and
randomly-injected **motion-artifact noise bursts**, rather than a real
downloaded PPG recording.

This is a learning/demonstration project, not a PhysioNet/Computing in
Cardiology Challenge submission, and no claim is made otherwise. The
pipeline (bandpass filtering, peak detection, HRV metric computation) is
standard and signal-source-agnostic — it would run unchanged on a real
PPG recording (e.g. a PhysioNet or wearable-device export) with a
different loader.

## What it does

1. **Signal generation** (`generate_ppg.py`) — 3 minutes of synthetic PPG
   at 100 Hz with HRV driven by a simulated 0.25 Hz respiration cycle,
   plus sensor noise and a few short motion-artifact bursts.
2. **Peak detection** (`peak_detect.py`) — bandpass filter (0.5–8 Hz) →
   adaptive-threshold peak picking → rejection of physiologically
   implausible intervals (catches most, not all, motion-artifact false
   positives — deliberately imperfect, see results below).
3. **HRV metrics** (`hrv_metrics.py`) — time-domain (mean HR, SDNN, RMSSD,
   pNN50) and frequency-domain (VLF/LF/HF band power via Welch PSD of the
   cubic-interpolated, resampled RR tachogram, LF/HF ratio).
4. **Pipeline + scoring** (`run_pipeline.py`) — runs everything, scores
   peak detection against ground truth, and compares HRV metrics computed
   from detected peaks vs. ground-truth peaks.

## Results (this run)

- Peak detector: **99.5% sensitivity, 98.5% precision** vs. ground truth
  (tolerance ±150 ms).
- HRV metrics, detected vs. reference:

  | Metric | Reference | Detected |
  |---|---:|---:|
  | Mean HR (bpm) | 67.9 | 68.6 |
  | SDNN (ms) | 58.4 | 82.1 |
  | RMSSD (ms) | 75.3 | 95.5 |

The detected SDNN/RMSSD are visibly inflated relative to ground truth —
and the tachogram plot (`outputs/rr_tachogram.png`) shows exactly why:
two motion-artifact bursts produce brief spurious interval spikes that
the plausibility filter doesn't fully catch, which HRV metrics (RMSSD
especially) are very sensitive to. This is included deliberately rather
than tuned away, because it's a realistic and well-known failure mode of
PPG-based HRV in the wearables literature (motion artifact is the
dominant PPG HRV error source), and a peak detector that silently
produces clean-looking but wrong HRV numbers would be a worse
demonstration than one that shows, and explains, its own error mode.

Full numbers in `outputs/metrics.json`.

## Run it

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py
```

Output (metrics + plots) is written to `outputs/`.

## Project structure

```
proj3_ppg_hrv/
├── README.md
├── requirements.txt
├── src/
│   ├── generate_ppg.py     # synthetic PPG generator (RSA-driven HRV + motion artifacts)
│   ├── peak_detect.py      # from-scratch systolic peak detector
│   ├── hrv_metrics.py      # time-domain + frequency-domain HRV metrics
│   └── run_pipeline.py     # end-to-end: generate -> detect -> score -> HRV -> plots
└── outputs/                # generated on run: metrics.json + plots
```
