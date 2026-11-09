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

## Run it

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py
```

Output (metrics + plots) is written to `outputs/`.

## Results (this run)

Numbers are regenerated live by the pipeline — `outputs/metrics.json`
is the authoritative record of every run. Current build (seed 7, 3 min
of 100 Hz synthetic PPG, 205 beats, 0.25 Hz RSA-modulated HRV, 3
motion-artifact bursts):

- Peak detector: **99.5% sensitivity, 100.0% precision** vs. ground
  truth (tolerance ±150 ms) — a single missed beat near an artifact burst.
- HRV metrics, detected vs reference (close agreement despite the one
  artifact-region miss):

  | Metric | Reference | Detected |
  |---|---:|---:|
  | Mean HR (bpm) | 68.37 | 68.37 |
  | SDNN (ms) | 62.32 | 59.55 |
  | RMSSD (ms) | 79.73 | 76.32 |
  | pNN50 (%) | 67.49 | 64.85 |

- Frequency domain: essentially all heart-rate power sits in the HF band
  (the simulated respiration is at 0.25 Hz, which falls inside the
  0.15–0.40 Hz HF band by definition), so the LF band — which in real
  recordings also carries baroreflex-mediated activity this generator
  does not model — is tiny, and the LF/HF ratio is correspondingly low.

Motion artifact remains the known dominant error mode of PPG-based HRV
in the wearables literature: in runs where an artifact bumps the
adaptive threshold, a missed detection or brief spurious interval
inflates RMSSD/SDNN sharply, exactly the failure mode real devices
must handle. The current artifact parameters happen to be handled
cleanly by the height threshold + plausibility filter (with one
boundary miss), so the run stays close to the reference — but the
tachogram plot (`outputs/rr_tachogram.png`) still marks the artifact
regions where that agreement is most fragile.

Full numbers in `outputs/metrics.json`.

## Project structure

```
project-3-ppg-hrv/
├── README.md
├── requirements.txt
├── src/
│   ├── generate_ppg.py     # synthetic PPG generator (RSA-driven HRV + motion artifacts)
│   ├── peak_detect.py      # from-scratch systolic peak detector
│   ├── hrv_metrics.py      # time-domain + frequency-domain HRV metrics
│   └── run_pipeline.py     # end-to-end: generate -> detect -> score -> HRV -> plots
├── tests/
│   └── test_pipeline.py    # fast unit tests of the core pieces
└── outputs/                # generated on run: metrics.json + plots
```
