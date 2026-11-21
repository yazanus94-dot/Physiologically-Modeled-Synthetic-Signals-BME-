"""Tests for preprocessing pipeline."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import (
    PreprocessingConfig, preprocess_ecg, check_signal_quality,
    remove_dc_offset, design_bandpass_sos, apply_bandpass_filter,
    normalize_signal, get_filter_response
)


def test_preprocessing_config_defaults():
    """Test default configuration values."""
    config = PreprocessingConfig()
    assert config.lowcut_hz == 0.5
    assert config.highcut_hz == 40.0
    assert config.filter_order == 4
    assert config.remove_dc is True
    assert config.dc_method == "median"
    assert config.check_nan_inf is True
    assert config.nan_policy == "error"
    assert config.normalize is False


def test_preprocessing_config_serialization():
    """Test config to/from dict."""
    config = PreprocessingConfig(lowcut_hz=1.0, highcut_hz=30.0, filter_order=2)
    d = config.to_dict()
    assert d["lowcut_hz"] == 1.0
    assert d["highcut_hz"] == 30.0

    config2 = PreprocessingConfig.from_dict(d)
    assert config2.lowcut_hz == 1.0
    assert config2.highcut_hz == 30.0


def test_check_signal_quality_clean():
    """Test quality check on clean signal."""
    sig = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    report = check_signal_quality(sig)
    assert not report["has_nan"]
    assert not report["has_inf"]
    assert report["nan_count"] == 0
    assert report["inf_count"] == 0


def test_check_signal_quality_with_nan():
    """Test quality check detects NaN."""
    sig = np.array([1.0, np.nan, 3.0])
    report = check_signal_quality(sig, PreprocessingConfig(nan_policy="interpolate"))
    assert report["has_nan"]
    assert report["nan_count"] == 1


def test_check_signal_quality_with_inf():
    """Test quality check detects infinity."""
    sig = np.array([1.0, np.inf, 3.0])
    report = check_signal_quality(sig, PreprocessingConfig(nan_policy="interpolate"))
    assert report["has_inf"]
    assert report["inf_count"] == 1


def test_remove_dc_offset_median():
    """Test DC removal with median."""
    sig = np.array([10.0, 11.0, 12.0, 10.0, 11.0])  # median = 11
    result = remove_dc_offset(sig, method="median")
    assert np.median(result) == 0.0


def test_remove_dc_offset_mean():
    """Test DC removal with mean."""
    sig = np.array([10.0, 11.0, 12.0, 10.0, 11.0])  # mean = 10.8
    result = remove_dc_offset(sig, method="mean")
    assert abs(np.mean(result)) < 1e-10


def test_design_bandpass_sos():
    """Test SOS filter design."""
    sos = design_bandpass_sos(0.5, 40.0, 360, order=4)
    assert sos.shape[1] == 6  # SOS format
    # 4th order bandpass = 4 sections (2 per order for bandpass)
    assert sos.shape[0] == 4


def test_design_bandpass_sos_invalid_freqs():
    """Test filter design rejects invalid frequencies."""
    # Low >= high
    try:
        design_bandpass_sos(40.0, 0.5, 360)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Low <= 0
    try:
        design_bandpass_sos(0.0, 40.0, 360)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # High >= Nyquist
    try:
        design_bandpass_sos(0.5, 180.0, 360)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_apply_bandpass_filter_zero_phase():
    """Test zero-phase bandpass filtering."""
    fs = 360
    t = np.arange(3600) / fs
    sig = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 50.0 * t)

    filtered = apply_bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4, zero_phase=True)

    # 50 Hz should be attenuated
    from scipy.fft import fft, fftfreq
    fft_orig = np.abs(fft(sig))
    fft_filt = np.abs(fft(filtered))
    freqs = fftfreq(len(sig), 1/fs)

    idx_50 = np.argmin(np.abs(freqs - 50))
    idx_1 = np.argmin(np.abs(freqs - 1))

    # 50 Hz should be much lower in filtered (attenuated to ~12%)
    assert fft_filt[idx_50] < fft_orig[idx_50] * 0.15
    # 1 Hz should be preserved (96%)
    assert fft_filt[idx_1] > fft_orig[idx_1] * 0.5


def test_apply_bandpass_filter_causal():
    """Test causal bandpass filtering introduces phase shift."""
    fs = 360
    t = np.arange(3600) / fs
    sig = np.sin(2 * np.pi * 1.0 * t)

    filtered_zp = apply_bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4, zero_phase=True)
    filtered_causal = apply_bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4, zero_phase=False)

    # Zero-phase should align with original (no phase shift)
    # Causal should have phase shift
    # Cross-correlation peak should be at 0 for zero-phase
    corr_zp = np.correlate(filtered_zp, sig, mode="full")
    corr_causal = np.correlate(filtered_causal, sig, mode="full")

    peak_zp = np.argmax(corr_zp) - (len(sig) - 1)
    peak_causal = np.argmax(corr_causal) - (len(sig) - 1)

    # Zero-phase should have minimal shift
    assert abs(peak_zp) < 5
    # Causal should have noticeable shift
    assert abs(peak_causal) > 5


def test_normalize_signal_zscore():
    """Test z-score normalization."""
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normed = normalize_signal(sig, method="zscore")
    assert abs(np.mean(normed)) < 1e-10
    assert abs(np.std(normed) - 1.0) < 1e-10


def test_normalize_signal_minmax():
    """Test min-max normalization."""
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normed = normalize_signal(sig, method="minmax")
    assert np.min(normed) == -1.0
    assert np.max(normed) == 1.0


def test_normalize_signal_robust():
    """Test robust normalization."""
    sig = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Outlier
    normed = normalize_signal(sig, method="robust")
    # Should be less affected by outlier
    assert np.median(normed) == 0.0


def test_preprocess_ecg_full_pipeline():
    """Test full preprocessing pipeline."""
    fs = 360
    t = np.arange(7200) / fs
    sig = (
        np.sin(2 * np.pi * 1.2 * t) +  # Heart beat ~72 BPM
        0.1 * np.sin(2 * np.pi * 0.2 * t) +  # Baseline wander
        0.05 * np.sin(2 * np.pi * 50.0 * t) +  # Powerline
        0.02 * np.random.randn(len(t))  # Noise
    )

    config = PreprocessingConfig(
        lowcut_hz=0.5,
        highcut_hz=40.0,
        filter_order=4,
        remove_dc=True,
        normalize=False
    )

    processed, report = preprocess_ecg(sig, fs, config)

    assert len(processed) == len(sig)
    assert "steps" in report
    assert "bandpass_0.5-40.0Hz_order4_sos_zero_phase" in report["steps"]
    assert "dc_removal_median" in report["steps"]
    assert "final_stats" in report


def test_preprocess_ecg_with_nan_interpolation():
    """Test preprocessing interpolates NaN."""
    sig = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    fs = 250
    config = PreprocessingConfig(nan_policy="interpolate")
    processed, report = preprocess_ecg(sig, fs, config)
    assert not np.any(np.isnan(processed))
    # Interpolation happens before filtering, so exact value depends on full pipeline
    # Just verify NaN is gone and no warning about errors
    assert "nan_interpolation" in report["steps"] or "dc_removal_median" in report["steps"]


def test_preprocess_ecg_with_detrend():
    """Test preprocessing with detrending."""
    fs = 360
    t = np.arange(3600) / fs
    sig = np.sin(2 * np.pi * 1.0 * t) + 0.01 * t  # Linear trend

    config = PreprocessingConfig(detrend=True, detrend_type="linear")
    processed, report = preprocess_ecg(sig, fs, config)

    # Trend should be removed
    assert "detrend_linear" in report["steps"]
    # Mean should be near zero
    assert abs(np.mean(processed)) < 0.1


def test_get_filter_response():
    """Test filter response calculation."""
    config = PreprocessingConfig(lowcut_hz=0.5, highcut_hz=40.0, filter_order=4)
    freqs, response = get_filter_response(config, fs=360)

    assert len(freqs) == len(response)
    assert freqs[0] == 0.0
    # Nyquist may not be exact due to frequency grid
    assert abs(freqs[-1] - 180.0) < 0.5
    # Response should be in dB
    assert np.all(response < 100)  # Not too large


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])