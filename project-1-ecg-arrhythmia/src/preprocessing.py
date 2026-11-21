"""
Reproducible ECG preprocessing pipeline.

Implements standardized preprocessing for ECG signals including:
- NaN/infinity checking
- DC offset removal
- Bandpass filtering (Butterworth, second-order sections, zero-phase)
- Configurable parameters with full documentation
"""
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfreqz
from scipy.stats import zscore


@dataclass
class PreprocessingConfig:
    """Configuration for ECG preprocessing pipeline.

    All parameters should be recorded for reproducibility.
    """
    # Bandpass filter
    lowcut_hz: float = 0.5
    highcut_hz: float = 40.0
    filter_order: int = 4
    filter_type: str = "butterworth"

    # DC removal
    remove_dc: bool = True
    dc_method: str = "median"  # "median", "mean", "highpass"

    # Quality checks
    check_nan_inf: bool = True
    nan_policy: str = "error"  # "error", "interpolate", "remove"

    # Normalization
    normalize: bool = False
    norm_method: str = "zscore"  # "zscore", "minmax", "robust"

    # Additional
    detrend: bool = False
    detrend_type: str = "linear"  # "linear", "constant"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreprocessingConfig":
        """Create from dictionary."""
        return cls(**d)

    def __str__(self) -> str:
        lines = ["Preprocessing Configuration:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


DEFAULT_CONFIG = PreprocessingConfig()


def check_signal_quality(
    signal: np.ndarray,
    config: Optional[PreprocessingConfig] = None
) -> Dict[str, Any]:
    """
    Check signal for NaN, infinity, and other quality issues.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal
    config : PreprocessingConfig, optional
        Configuration for handling issues

    Returns
    -------
    dict
        Quality report with keys: has_nan, has_inf, nan_count, inf_count, etc.
    """
    if config is None:
        config = DEFAULT_CONFIG

    report = {
        "length": len(signal),
        "has_nan": bool(np.any(np.isnan(signal))),
        "has_inf": bool(np.any(np.isinf(signal))),
        "nan_count": int(np.sum(np.isnan(signal))),
        "inf_count": int(np.sum(np.isinf(signal))),
        "min": float(np.nanmin(signal)) if not np.all(np.isnan(signal)) else np.nan,
        "max": float(np.nanmax(signal)) if not np.all(np.isnan(signal)) else np.nan,
        "mean": float(np.nanmean(signal)) if not np.all(np.isnan(signal)) else np.nan,
        "std": float(np.nanstd(signal)) if not np.all(np.isnan(signal)) else np.nan,
    }

    if report["has_nan"] or report["has_inf"]:
        msg = f"Signal quality issues: NaN={report['nan_count']}, Inf={report['inf_count']}"
        if config.nan_policy == "error":
            raise ValueError(msg)
        elif config.nan_policy == "interpolate":
            warnings.warn(msg + " - interpolating")
        elif config.nan_policy == "remove":
            warnings.warn(msg + " - removing")

    return report


def remove_dc_offset(
    signal: np.ndarray,
    method: str = "median"
) -> np.ndarray:
    """
    Remove DC offset from signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str
        Method to estimate DC: "median" (robust), "mean", or "highpass"

    Returns
    -------
    np.ndarray
        DC-removed signal
    """
    if method == "median":
        dc = np.nanmedian(signal)
    elif method == "mean":
        dc = np.nanmean(signal)
    elif method == "highpass":
        # This would be done via filtering, not here
        return signal
    else:
        raise ValueError(f"Unknown DC removal method: {method}")

    return signal - dc


def design_bandpass_sos(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Design a Butterworth bandpass filter as second-order sections (SOS).

    Using SOS format provides better numerical stability than (b, a) format.

    Parameters
    ----------
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order (must be even for bandpass)

    Returns
    -------
    sos : np.ndarray
        Second-order sections array of shape (n_sections, 6)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low <= 0 or high >= 1 or low >= high:
        raise ValueError(f"Invalid cutoff frequencies: low={lowcut}, high={highcut}, fs={fs}")

    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def apply_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
    zero_phase: bool = True
) -> np.ndarray:
    """
    Apply bandpass filter to ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal
    fs : float
        Sampling frequency in Hz
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    order : int
        Filter order
    zero_phase : bool
        If True, use zero-phase filtering (filtfilt/sosfiltfilt).
        If False, use causal filtering (introduces phase delay).

    Returns
    -------
    np.ndarray
        Filtered signal
    """
    sos = design_bandpass_sos(lowcut, highcut, fs, order)

    if zero_phase:
        # Zero-phase filtering - no phase distortion
        filtered = sosfiltfilt(sos, signal, padlen=min(len(signal) - 1, 3 * order * 2))
    else:
        # Causal filtering - introduces phase delay
        from scipy.signal import sosfilt
        filtered = sosfilt(sos, signal)

    return filtered


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore"
) -> np.ndarray:
    """
    Normalize signal amplitude.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str
        Normalization method: "zscore", "minmax", "robust"

    Returns
    -------
    np.ndarray
        Normalized signal
    """
    if method == "zscore":
        return zscore(signal, nan_policy="omit")
    elif method == "minmax":
        smin, smax = np.nanmin(signal), np.nanmax(signal)
        if smax > smin:
            return 2 * (signal - smin) / (smax - smin) - 1
        return signal
    elif method == "robust":
        median = np.nanmedian(signal)
        mad = np.nanmedian(np.abs(signal - median))
        if mad > 0:
            return (signal - median) / (1.4826 * mad)
        return signal
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_ecg(
    signal: np.ndarray,
    fs: float,
    config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Full ECG preprocessing pipeline.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal
    fs : float
        Sampling frequency in Hz
    config : PreprocessingConfig, optional
        Preprocessing configuration. Uses defaults if None.

    Returns
    -------
    processed_signal : np.ndarray
        Preprocessed ECG signal
    report : dict
        Processing report with all parameters and quality metrics
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Make a copy to avoid modifying original
    signal = np.asarray(signal, dtype=float).copy()
    original_length = len(signal)

    report = {
        "config": config.to_dict(),
        "fs": fs,
        "original_length": original_length,
        "steps": []
    }

    # Step 1: Quality check
    if config.check_nan_inf:
        quality = check_signal_quality(signal, config)
        report["quality_check"] = quality
        report["steps"].append("quality_check")

        # Handle NaN/Inf if needed
        if quality["has_nan"] and config.nan_policy == "interpolate":
            # Linear interpolation for NaN values
            nan_mask = np.isnan(signal)
            if not np.all(nan_mask):
                indices = np.arange(len(signal))
                signal[nan_mask] = np.interp(
                    indices[nan_mask], indices[~nan_mask], signal[~nan_mask]
                )
            report["steps"].append("nan_interpolation")

        if quality["has_inf"] and config.nan_policy == "interpolate":
            inf_mask = np.isinf(signal)
            if not np.all(inf_mask):
                indices = np.arange(len(signal))
                signal[inf_mask] = np.interp(
                    indices[inf_mask], indices[~inf_mask], signal[~inf_mask]
                )
            report["steps"].append("inf_interpolation")

    # Step 2: Remove DC offset
    if config.remove_dc:
        signal = remove_dc_offset(signal, config.dc_method)
        report["steps"].append(f"dc_removal_{config.dc_method}")

    # Step 3: Detrend
    if config.detrend:
        from scipy.signal import detrend
        signal = detrend(signal, type=config.detrend_type)
        report["steps"].append(f"detrend_{config.detrend_type}")

    # Step 4: Bandpass filter
    signal = apply_bandpass_filter(
        signal, fs,
        lowcut=config.lowcut_hz,
        highcut=config.highcut_hz,
        order=config.filter_order,
        zero_phase=True  # Always zero-phase for analysis
    )
    report["steps"].append(
        f"bandpass_{config.lowcut_hz}-{config.highcut_hz}Hz_order{config.filter_order}_sos_zero_phase"
    )

    # Step 5: Normalize
    if config.normalize:
        signal = normalize_signal(signal, config.norm_method)
        report["steps"].append(f"normalize_{config.norm_method}")

    # Final stats
    report["final_stats"] = {
        "length": len(signal),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
    }

    return signal, report


def get_filter_response(
    config: PreprocessingConfig,
    fs: float,
    n_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get frequency response of the configured filter.

    Returns
    -------
    freqs : np.ndarray
        Frequencies in Hz
    response : np.ndarray
        Magnitude response in dB
    """
    sos = design_bandpass_sos(
        config.lowcut_hz, config.highcut_hz, fs, config.filter_order
    )
    w, h = sosfreqz(sos, worN=n_points, fs=fs)
    return w, 20 * np.log10(np.abs(h) + 1e-12)


if __name__ == "__main__":
    # Quick test
    fs = 360
    t = np.arange(3600) / fs
    # Synthetic ECG-like signal
    signal = (
        1.0 * np.sin(2 * np.pi * 1.2 * t) +  # ~72 BPM heart rate
        0.1 * np.sin(2 * np.pi * 0.2 * t) +  # Baseline wander
        0.05 * np.sin(2 * np.pi * 50 * t) +  # Powerline
        0.02 * np.random.randn(len(t))       # Noise
    )

    config = PreprocessingConfig(
        lowcut_hz=0.5,
        highcut_hz=40.0,
        filter_order=4,
        remove_dc=True,
        normalize=False
    )

    processed, report = preprocess_ecg(signal, fs, config)
    print("Preprocessing report:")
    for key, value in report.items():
        if key != "config":
            print(f"  {key}: {value}")
    print(f"\nConfig:\n{config}")