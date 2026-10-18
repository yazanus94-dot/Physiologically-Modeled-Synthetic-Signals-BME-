"""
Generates a clean synthetic ECG plus a noisy version with three realistic,
independently-controllable artifact sources common in real ambulatory ECG
recordings:
  - baseline wander (low-frequency drift from respiration/movement)
  - powerline interference (50/60 Hz mains pickup)
  - EMG/muscle noise (broadband high-frequency noise)

Having the clean ground-truth signal lets the denoising pipeline be scored
objectively (SNR improvement, RMSE), which isn't possible with real
recordings (no clean reference exists for real ambulatory ECG) — that's
the deliberate reason this project uses synthetic data rather than real
PhysioNet records in an environment without internet access to download
them. See README.md for the honest framing.
"""
import numpy as np


def _gaussian(t, mu, sigma, a):
    return a * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


def _synthesize_clean_record(duration_s, fs, mean_hr_bpm, seed):
    """Self-contained sum-of-Gaussians clean ECG generator (normal beats
    only — this project is about denoising, not beat classification, so it
    doesn't need the PVC/other morphologies project 1 uses)."""
    rng = np.random.default_rng(seed)
    waves = [
        (-0.20, 0.025, 0.15), (-0.05, 0.010, -0.10), (0.00, 0.012, 1.00),
        (0.05, 0.010, -0.20), (0.30, 0.060, 0.30),
    ]
    chunks, r_peaks, labels = [], [], []
    t_cursor, sample_cursor = 0.0, 0
    while t_cursor < duration_s:
        rr = max(0.4, 60.0 / mean_hr_bpm + rng.normal(0, 0.02))
        n = int(round(rr * fs))
        t = np.linspace(-0.5 * rr, 0.5 * rr, n)
        beat = np.zeros_like(t)
        for mu, sigma, amp in waves:
            beat += _gaussian(t, mu, sigma, amp)
        r_in_beat = int(np.argmax(np.abs(beat[: n // 2 + n // 10])))
        chunks.append(beat)
        r_peaks.append(sample_cursor + r_in_beat)
        labels.append("N")
        sample_cursor += n
        t_cursor += rr
    return np.concatenate(chunks), fs, np.array(r_peaks), labels


def make_clean_and_noisy(duration_s=30, fs=250, seed=3,
                          baseline_amp=0.25, powerline_amp=0.08,
                          powerline_hz=50.0, emg_std=0.05):
    clean, fs, r_peaks, labels = _synthesize_clean_record(duration_s, fs, mean_hr_bpm=70, seed=seed)
    n = len(clean)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed + 1)

    baseline = baseline_amp * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi)) \
        + 0.5 * baseline_amp * np.sin(2 * np.pi * 0.08 * t)
    powerline = powerline_amp * np.sin(2 * np.pi * powerline_hz * t)
    emg = rng.normal(0, emg_std, size=n)

    noisy = clean + baseline + powerline + emg
    return clean, noisy, fs, r_peaks, labels, {
        "baseline": baseline, "powerline": powerline, "emg": emg,
    }
