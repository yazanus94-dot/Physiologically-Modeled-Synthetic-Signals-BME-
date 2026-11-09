"""
Four denoising approaches compared on the same noisy ECG signal:

  1. Median-filter baseline removal (classic clinical-ECG baseline wander
     correction: estimate baseline with a wide median filter, subtract it)
  2. Butterworth bandpass filtering (standard 0.5-40 Hz ECG passband)
  3. IIR notch filter targeted at the powerline frequency
  4. FFT-domain soft-thresholding denoiser (frequency-domain analogue of
     wavelet soft-thresholding denoising, implemented without needing an
     external wavelet library)

And a combined pipeline (1 -> 3 -> 2) representing what a real clinical
ECG front-end preprocessing chain typically looks like.
"""
import numpy as np
from scipy.signal import butter, filtfilt, medfilt, iirnotch


def median_filter_baseline_removal(signal, fs, window_s=0.6):
    window = int(round(window_s * fs))
    if window % 2 == 0:
        window += 1
    baseline = medfilt(signal, kernel_size=window)
    return signal - baseline, baseline


def bandpass_filter(signal, fs, low=0.5, high=40.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq=50.0, quality=30.0):
    b, a = iirnotch(freq / (fs / 2), quality)
    return filtfilt(b, a, signal)


def fft_soft_threshold_denoise(signal, fs, keep_band=(0.5, 40.0), threshold_fraction=0.08):
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    out_of_band = (freqs < keep_band[0]) | (freqs > keep_band[1])
    spectrum[out_of_band] = 0.0

    mag = np.abs(spectrum)
    thresh = threshold_fraction * mag.max()
    magnitude_reduced = np.maximum(mag - thresh, 0)
    phase = np.angle(spectrum)
    spectrum_denoised = magnitude_reduced * np.exp(1j * phase)

    return np.fft.irfft(spectrum_denoised, n=n)


def combined_pipeline(signal, fs, powerline_hz=50.0):
    step1, baseline = median_filter_baseline_removal(signal, fs)
    step2 = notch_filter(step1, fs, freq=powerline_hz)
    step3 = bandpass_filter(step2, fs)
    return step3


def snr_db(clean, estimate):
    noise = clean - estimate
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def rmse(clean, estimate):
    return float(np.sqrt(np.mean((clean - estimate) ** 2)))
