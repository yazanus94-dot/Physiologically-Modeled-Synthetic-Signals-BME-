"""
HRV metrics from pulse-to-pulse interval (tachogram) series.

Time domain — Task Force (ESC/NASPE) conventions:
  Mean HR, SDNN, RMSSD, pNN50
Frequency domain:
  VLF/LF/HF band power via Welch PSD of the cubic-interpolated,
  uniformly resampled tachogram (4 Hz), and the LF/HF ratio.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import welch

WELCH_FS = 4.0  # Hz, resampled tachogram rate


def _clean_tachogram(peaks, fs):
    peaks = np.sort(np.asarray(peaks, dtype=int))
    itis_s = np.diff(peaks) / fs  # intervals in seconds
    t_mid = (peaks[:-1] + peaks[1:]) / 2.0 / fs
    return itis_s, t_mid


def time_domain_metrics(itis_ms):
    """SDNN / RMSSD / pNN50 from an interval series in milliseconds."""
    itis_ms = np.asarray(itis_ms, dtype=float)
    if len(itis_ms) < 2:
        return {"mean_hr_bpm": float("nan"), "sdnn_ms": float("nan"),
                "rmssd_ms": float("nan"), "pnn50_pct": float("nan")}
    mean_hr = 60.0 / np.mean(itis_ms / 1000.0)
    sdnn = float(np.std(itis_ms, ddof=1))
    diffs = np.diff(itis_ms)
    rmssd = float(np.sqrt(np.mean(diffs * diffs)))
    pnn50 = float(np.mean(np.abs(diffs) > 50.0)) * 100.0
    return {"mean_hr_bpm": round(float(mean_hr), 2),
            "sdnn_ms": round(sdnn, 2),
            "rmssd_ms": round(rmssd, 2),
            "pnn50_pct": round(pnn50, 2)}


def frequency_domain_metrics(peaks, fs):
    """VLF/LF/HF power and LF/HF ratio from a peak series via Welch PSD."""
    return _frequency_domain_metrics_from_peaks(peaks, fs)


def _frequency_domain_metrics_from_peaks(peaks, fs):
    itis_s, t_mid = _clean_tachogram(peaks, fs)
    if len(itis_s) < 16:  # LSQ interpolation of such short series is not meaningful
        return {"vlf_power": float("nan"), "lf_power": float("nan"),
                "hf_power": float("nan"), "lf_hf_ratio": float("nan")}

    # cubic interpolation to a uniformly sampled tachogram, per Task Force
    cs = CubicSpline(t_mid, itis_s * 1000.0)  # ms
    t_uniform = np.arange(t_mid[0], t_mid[-1], 1.0 / WELCH_FS)
    if len(t_uniform) < 2:
        return {"vlf_power": float("nan"), "lf_power": float("nan"),
                "hf_power": float("nan"), "lf_hf_ratio": float("nan")}
    detrended = cs(t_uniform)
    detrended = detrended - np.mean(detrended)

    freqs, psd = welch(detrended, fs=WELCH_FS, nperseg=min(256, len(detrended)))
    vlf = float(np.trapezoid(psd[(freqs >= 0.003) & (freqs < 0.04)],
                             freqs[(freqs >= 0.003) & (freqs < 0.04)]))
    lf = float(np.trapezoid(psd[(freqs >= 0.04) & (freqs < 0.15)],
                            freqs[(freqs >= 0.04) & (freqs < 0.15)]))
    hf = float(np.trapezoid(psd[(freqs >= 0.15) & (freqs <= 0.40)],
                            freqs[(freqs >= 0.15) & (freqs <= 0.40)]))
    ratio = lf / hf if hf > 0 else float("nan")
    return {"vlf_power": round(vlf, 3), "lf_power": round(lf, 3),
            "hf_power": round(hf, 3), "lf_hf_ratio": round(ratio, 3)}
