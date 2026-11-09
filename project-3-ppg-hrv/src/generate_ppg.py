"""
Physiologically-motivated synthetic PPG (photoplethysmogram) generator.

3 minutes of PPG at 100 Hz with heart rate variability driven by a
simulated 0.25 Hz respiration cycle (respiratory sinus arrhythmia, the
dominant physiological driver of short-term HRV), asymmetric
double-Gaussian pulses (systolic peak + dicrotic notch), sensor noise,
and a few short motion-artifact bursts.
"""
import numpy as np

DEFAULT_FS = 100


def _pulse_waveform(fs, pulse_shape=(0.30, 0.16, 1.00, 0.55, 0.09, 0.35)):
    """Asymmetric double-Gaussian pulse: systolic peak + dicrotic bulk.

    pulse_shape = (t_syst, sigma_syst, a_syst, t_dicrot, sigma_dicrot, a_dicrot)
    returned over a 1.5 s window; each pulse is scaled/positioned so its
    systolic peak lands at pulse_start + t_syst.
    """
    t_syst, sig_s, a_syst, t_d, sig_d, a_d = pulse_shape
    window = np.linspace(-0.35, 1.15, int(round(1.5 * fs)))
    pulse = (a_syst * np.exp(-((window - t_syst) ** 2) / (2 * sig_s ** 2))
             + a_d * np.exp(-((window - t_d) ** 2) / (2 * sig_d ** 2)))
    return window, pulse


def generate_ppg(duration_s=180.0, fs=DEFAULT_FS, mean_hr_bpm=68.0,
                 seed=7, rsa_depth=0.10, resp_hz=0.25,
                 sensor_noise_std=0.012, motion_bursts=3, noise=True):
    """Generate the synthetic PPG signal.

    RSA model: the heart period (RR interval) oscillates with the
    respiratory cycle — heart rate rises on inspiration. RR(t) is built
    beat by beat, sampled at each beat start, so the tachogram itself
    is the RSA signal.

    Returns
    -------
    ppg : np.ndarray
    t : np.ndarray       time vector (s)
    peaks_true : np.ndarray  sample indices of true systolic peaks
    fs : int
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    ppg = np.zeros(n)

    window, pulse = _pulse_waveform(fs)
    pk_off = int(round((0.30 + 0.35) * fs))  # pulse[peak] index for 0.30 s systolic

    rr0 = 60.0 / mean_hr_bpm

    t_beat = 0.0
    peaks = []
    while t_beat < duration_s:
        # RSA: RR oscillates ±rsa_depth with the breathing cycle
        rr = rr0 * (1.0 - rsa_depth * np.sin(2 * np.pi * resp_hz * t_beat))
        rr += rng.normal(0.0, 0.005)   # beat-to-beat jitter

        start = int(round(t_beat * fs))
        j0 = start - pk_off
        seg_len = min(len(pulse), n - j0)
        if j0 >= 0 and seg_len > 0:
            ppg[j0:j0 + seg_len] += pulse[:seg_len]

        peak_idx = j0 + pk_off
        if 0 <= peak_idx < n:
            peaks.append(peak_idx)

        t_beat += rr

    # --- sensor noise ---
    if noise:
        ppg += rng.normal(0.0, sensor_noise_std, size=n)

        # --- motion-artifact bursts: short segments of low-frequency wiggle ---
        candidates = [0.25, 0.55, 0.82]
        for frac in candidates[:motion_bursts]:
            c0 = int(frac * n)
            dur = int(rng.uniform(0.4, 0.9) * fs)
            lo = max(0, c0)
            hi = min(n, c0 + dur)
            if hi <= lo:
                continue
            t_burst = np.arange(hi - lo) / fs
            amp = rng.uniform(0.45, 0.75)
            phase0 = rng.uniform(0, 2 * np.pi)
            artifact = amp * (0.6 * np.sin(2 * np.pi * 2.5 * t_burst + phase0)
                              + 0.4 * np.sin(2 * np.pi * 1.1 * t_burst + 0.7 * phase0))
            drift_amp = rng.uniform(0.2, 0.4) * np.linspace(1, 0, hi - lo)
            ppg[lo:hi] += artifact * drift_amp

    t = np.arange(n) / fs
    return ppg, t, np.asarray(peaks, dtype=int), fs
