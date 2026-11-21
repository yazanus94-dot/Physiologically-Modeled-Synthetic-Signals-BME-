"""Tests for the real NSTDB experiment modules.

These cover the pieces added for the PhysioNet Noise Stress Test Database
pipeline: noise-record discovery/parsing, the SNR mixer (which must hit
its target), DC-removed scoring, and the real-experiment runner's wiring.

Nothing here downloads anything: the loader tests exercise pure parsing
and the mixer tests are synthetic by construction.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from noise_mixer import (align_to_common, mix_to_snr,  # noqa: E402
                         scale_noise_to_snr, signal_power)
from load_real_noise import _category_for, parse_noise_record_names  # noqa: E402
from evaluate_denoising import compute_segment_metrics, _zero_mean  # noqa: E402


# ---------------------------------------------------------------------------
# Noise-record discovery / parsing
# ---------------------------------------------------------------------------

RECORD_DOC = ("The database contains noise records (`bw', `em', and `ma') "
              "that may be added to clean ECG recordings at various "
              "signal-to-noise ratios.")


def test_parse_noise_record_names():
    """The parser extracts PhysioNet's quoted noise-record abbreviations."""
    names = parse_noise_record_names(RECORD_DOC)
    assert names == ["bw", "em", "ma"]


def test_parse_noise_record_names_requires_the_documented_sentence():
    with pytest.raises(ValueError):
        parse_noise_record_names("no noise records sentence here")


def test_category_for_maps_physionet_suffixes():
    assert _category_for("bw") == "baseline_wander"
    assert _category_for("ma") == "muscle_artifact"
    assert _category_for("em") == "electrode_motion"


def test_category_for_rejects_unknown_suffix():
    with pytest.raises(ValueError):
        _category_for("zz")


def test_parsing_is_pure_and_reproducible():
    a = parse_noise_record_names(RECORD_DOC)
    b = parse_noise_record_names(RECORD_DOC)
    assert a == b


# ---------------------------------------------------------------------------
# SNR mixer — the core claim of the real experiment
# ---------------------------------------------------------------------------

def _clean(n=8000, fs=250, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    # quasi-ECG: dominant 1.2 Hz deflection + small structure
    return (1.0 * np.sin(2 * np.pi * 1.2 * t)
            + 0.2 * np.sin(2 * np.pi * 6.0 * t)
            + 0.01 * rng.standard_normal(n))


def test_signal_power_matches_mean_square():
    x = np.array([1.0, -1.0, 2.0, 0.0])
    assert np.isclose(signal_power(x), np.mean(x ** 2))


def test_scale_noise_to_snr_hits_target_exactly():
    clean = _clean()
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(clean.size)
    for target in (-6.0, 0.0, 6.0, 12.0, 18.0, 24.0):
        scaled, achieved = scale_noise_to_snr(clean, noise, target)
        assert achieved == pytest.approx(target, abs=1e-9), (target, achieved)
        # and the reported value really is 10*log10(P_clean/P_noise)
        recomputed = 10 * np.log10(signal_power(clean) /
                                   signal_power(scaled))
        assert recomputed == pytest.approx(target, abs=1e-9)


def test_scale_noise_preserves_noise_shape():
    """Scaling must be a pure gain — the noise waveform itself is untouched."""
    clean = _clean()
    noise = np.random.default_rng(2).standard_normal(clean.size)
    scaled, _ = scale_noise_to_snr(clean, noise, 10.0)
    # up to a positive scalar factor the direction is identical
    gain = scaled / noise
    assert np.allclose(gain, gain[0], rtol=1e-9)


def test_scale_noise_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        scale_noise_to_snr(np.zeros(10), np.zeros(11), 6.0)


def test_scale_noise_rejects_nonfinite_target():
    clean = _clean()
    noise = np.random.default_rng(2).standard_normal(clean.size)
    with pytest.raises(ValueError):
        scale_noise_to_snr(clean, noise, np.nan)


def test_scale_noise_rejects_zero_power_signal():
    with pytest.raises(ValueError):
        scale_noise_to_snr(np.zeros(100), np.ones(100), 6.0)


def test_mix_to_snr_returns_mixture_at_target():
    clean = _clean()
    noise = np.random.default_rng(3).standard_normal(clean.size)
    for target in (-6.0, 6.0, 18.0):
        mixed, achieved = mix_to_snr(clean, noise, target, fs_clean=250,
                                     fs_noise=250, fs_out=250)
        assert mixed.shape == clean.shape
        assert np.isclose(achieved, target, atol=1e-6)
        # and the reported value really is the SNR of the returned mixture
        recomputed = 10 * np.log10(signal_power(clean) /
                                   signal_power(mixed - clean))
        assert np.isclose(recomputed, achieved, atol=1e-6)


def test_mix_to_snr_accepts_differing_sample_rates():
    """Resampling path: noise at 360 Hz against ECG at 250 Hz."""
    clean = _clean(n=7500, fs=250)
    noise = np.random.default_rng(4).standard_normal(10800)  # 360 Hz * 30 s
    mixed, achieved = mix_to_snr(clean, noise, 12.0, fs_clean=250,
                                 fs_noise=360, fs_out=250)
    assert mixed.shape == clean.shape
    assert np.isclose(achieved, 12.0, atol=1e-6)


def test_align_to_common_trims_to_shorter():
    a = np.arange(100, dtype=float)
    b = np.arange(80, dtype=float)
    aa, bb = align_to_common(a, b)
    assert aa.size == bb.size == 80
    assert np.array_equal(aa, a[:80])


def test_zero_snr_means_noise_equals_signal_power():
    clean = _clean()
    noise = np.random.default_rng(5).standard_normal(clean.size)
    scaled, achieved = scale_noise_to_snr(clean, noise, 0.0)
    assert achieved == pytest.approx(0.0, abs=1e-9)
    assert np.isclose(signal_power(scaled), signal_power(clean), rtol=1e-9)


# ---------------------------------------------------------------------------
# DC-removed scoring (regression for the offset defect)
# ---------------------------------------------------------------------------

def test_compute_segment_metrics_zero_means_both_sides():
    fs = 250
    t = np.arange(30 * fs) / fs
    clean = np.sin(2 * np.pi * 1.1 * t) - 0.33
    estimate = clean - clean.mean()          # what a high-pass produces
    m = compute_segment_metrics(clean, estimate, fs)
    assert m["rmse"] < 1e-9
    assert m["snr_db"] > 30
    assert abs(m["pearson"] - 1.0) < 1e-9


def test_zero_mean_helper_leaves_shape_and_differences():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    z = _zero_mean(x)
    assert z.shape == x.shape
    assert abs(z.mean()) < 1e-12
    assert np.allclose(np.diff(z), np.diff(x))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        compute_segment_metrics(np.zeros(10), np.zeros(11), 250)


# ---------------------------------------------------------------------------
# Real-experiment outputs: coverage and SNR verification
# ---------------------------------------------------------------------------

REPO = os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__))))
SEGMENTS_CSV = os.path.join(REPO, "outputs", "real", "results_segments.csv")
SUMMARY_CSV = os.path.join(REPO, "outputs", "real", "results_summary.csv")


def _real_segments():
    import csv
    if not os.path.exists(SEGMENTS_CSV):
        pytest.skip("real NSTDB segment results not present (run the "
                    "experiment first)")
    with open(SEGMENTS_CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def test_real_segments_have_at_least_five_per_cell():
    """Every (noise x SNR) cell needs >= 5 independent segments.

    Regression: the first real run produced 1 segment per cell (`--segments`
    defaulted to 1), so the summary average was a single noisy sample.  With
    5 segments the mean gain is a 5-sample average with reproducible seeds.
    """
    rows = _real_segments()
    cells = {}
    for r in rows:
        cells.setdefault((r["noise_category"], r["target_snr_db"]),
                         set()).add(r["segment"])
    assert cells, "results_segments.csv is empty"
    seg_counts = {c: len(v) for c, v in cells.items()}
    assert all(n >= 5 for n in seg_counts.values()), seg_counts
    assert len(cells) >= 3 * 6, "all 3 noise types x 6 SNR levels expected"


def test_real_segments_cover_all_snr_levels_and_noise_types():
    rows = _real_segments()
    snrs = {r["target_snr_db"] for r in rows}
    assert snrs == {"-6.0", "0.0", "6.0", "12.0", "18.0", "24.0"}, snrs
    cats = {r["noise_category"] for r in rows}
    assert cats == {"baseline_wander", "electrode_motion", "muscle_artifact"}


def test_real_achieved_snr_matches_target():
    """The mixer's achieved SNR must equal the target to ~1e-9 dB."""
    rows = _real_segments()
    worst = max(abs(float(r["achieved_snr_db"]) - float(r["target_snr_db"]))
                for r in rows)
    assert worst < 1e-6, worst


def test_real_summary_has_a_row_per_cell_and_method():
    import csv
    if not os.path.exists(SUMMARY_CSV):
        pytest.skip("results_summary.csv not present")
    with open(SUMMARY_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cells = {(r["noise_category"], r["target_snr_db"]) for r in rows}
    assert len(cells) >= 3 * 6
