"""
Physiologically-informed synthetic ECG generator.

Ten minutes of synthetic ECG at 250 Hz with randomly interspersed
Normal (N), PVC (V), and supraventricular/other (S) beats, each with a
distinct sum-of-Gaussians morphology (in the style of ECGSYN), plus
realistic baseline wander, 50 Hz powerline interference, and EMG-like noise.

Why synthetic? This project was built without internet access to PhysioNet,
so a generator is used instead of real MIT-BIH records — with the honest
framing that any reported performance applies to this synthetic dataset
only. The R-peak detection / classification pipeline is written against the
same output interface that `wfdb` helpers in real PhysioNet loaders return;
see load_real_mitbih_record() for the exact calls needed to swap in
real data.
"""
import os

import numpy as np

DEFAULT_FS = 250

# --- morphologies: (offset from R peak in s, gaussian sigma in s, amplitude) ---
# Normal beat: P wave, QRS complex, T wave
_WAVES_N = [
    (-0.20, 0.025, 0.15),   # P
    (-0.05, 0.010, -0.10),  # Q
    (0.000, 0.012, 1.00),   # R
    (0.050, 0.010, -0.20),  # S
    (0.300, 0.060, 0.30),   # T
]
# PVC beat: wide, tall, bizarre-looking, with no P wave
_WAVES_V = [
    (-0.060, 0.030, -0.25),
    (0.000, 0.045, 1.20),   # wide R
    (0.120, 0.040, -0.45),  # large discordant T
    (0.350, 0.070, 0.35),
]
# Supraventricular/other: ectopic P' before a narrow QRS
_WAVES_S = [
    (-0.160, 0.020, 0.22),  # P'
    (-0.050, 0.008, -0.12),
    (0.000, 0.011, 0.95),   # narrow R
    (0.050, 0.009, -0.22),
    (0.280, 0.055, 0.28),
]

_WAVES = {"N": _WAVES_N, "V": _WAVES_V, "S": _WAVES_S}

_HALF_W = 0.45  # seconds of signal kept on each side of the R peak


def _gaussian(t, mu, sigma, a):
    return a * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


def _beat_waveform(waves, fs):
    n = int(round(2 * _HALF_W * fs))
    t = np.linspace(-_HALF_W, _HALF_W, n)
    beat = np.zeros_like(t)
    for mu, sigma, amp in waves:
        beat += _gaussian(t, mu, sigma, amp)
    peak_idx = int(np.argmax(beat))
    return beat, peak_idx


def generate_ecg(duration_s=600.0, fs=DEFAULT_FS, mean_hr_bpm=70.0, seed=42,
                 p_beat=None, noise=True):
    """Generate the synthetic ECG.

    Returns
    -------
    signal : np.ndarray   the (optionally noisy) ECG in arbitrary mV
    r_peaks_true : np.ndarray   sample indices of the true R peaks
    labels : list[str]   one beat label ("N"/"V"/"S") per true peak
    fs : int
    """
    if p_beat is None:
        p_beat = {"N": 0.72, "V": 0.14, "S": 0.14}
    rng = np.random.default_rng(seed)
    beats = {cls: _beat_waveform(waves, fs) for cls, waves in _WAVES.items()}

    chunk = int(duration_s * fs)
    signal = np.zeros(chunk)
    r_peaks, labels = [], []
    classes, probs = zip(*p_beat.items())

    lead = int(round(_HALF_W * fs))  # beat array grows backward to its R peak
    cursor = lead                    # first R peak placed one lead-in in
    pending_pause = 1.0              # multiplier on next RR (compensatory pause after PVC)
    while cursor < chunk:
        cls = classes[int(rng.choice(len(classes), p=probs))]
        rr0 = 60.0 / mean_hr_bpm
        if cls == "S":
            rr = rr0 * 0.72            # premature supraventricular beat
        elif cls == "V":
            rr = rr0 * 0.80            # premature ventricular beat
        elif cls == "N" and labels and labels[-1] == "V":
            rr = rr0 * pending_pause   # compensatory pause after a PVC
        else:
            rr = rr0
        rr += rng.normal(0.0, 0.020)   # natural heart-rate variability
        rr = max(0.30, rr)

        r_idx = cursor
        r_peaks.append(r_idx)
        labels.append(cls)

        # place the beat additively; overlapping the next beat's window is
        # intentional and mimics T-wave/QRS co-occurrence in real ECG
        j0 = r_idx - lead
        beat, _peak_idx = beats[cls]
        n = min(len(beat), chunk - j0)
        if n > 0:
            signal[j0:j0 + n] += beat[:n]

        cursor += int(round(rr * fs))

        if cls == "V":
            pending_pause = rng.uniform(1.10, 1.25)  # compensatory pause
        else:
            pending_pause = 1.0

    if noise:
        t = np.arange(chunk) / fs
        baseline = 0.12 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
        baseline += 0.06 * np.sin(2 * np.pi * 0.08 * t)
        powerline = 0.05 * np.sin(2 * np.pi * 50.0 * t)
        emg = rng.normal(0.0, 0.035, size=chunk)
        signal = signal + baseline + powerline + emg

    return signal, np.array(r_peaks), labels, fs


def load_real_mitbih_record(data_path, record_name="100", ann_ext="atr"):
    """Optional helper to swap in a real PhysioNet MIT-BIH record.

    Requires the `wfdb` package (see requirements.txt). Returns the same
    (signal, r_peaks_true, labels, fs) tuple as generate_ecg.
    """
    import wfdb  # optional dependency, not required for the synthetic pipeline

    record = wfdb.rdrecord(os.path.join(data_path, record_name))
    ann = wfdb.rdann(os.path.join(data_path, record_name), ann_ext)
    signal = record.p_signal[:, 0]  # channel 0 = MLII, the usual analysis lead
    fs = record.fs
    symbols = set(ann.symbol)
    symbol_ok = [i for i, s in enumerate(ann.symbol) if s in symbols]
    return signal, ann.sample[symbol_ok], [ann.symbol[i] for i in symbol_ok], fs
