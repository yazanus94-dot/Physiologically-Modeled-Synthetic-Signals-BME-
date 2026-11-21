"""Tests for AAMI mapping and beat segmentation."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aami import (
    map_to_aami_class, filter_aami_beats,
    segment_beats_annotation_centered, count_beats_per_class,
    get_aami_classes, AAMI_MAPPING, EXCLUDED_SYMBOLS,
    is_beat_symbol, BEAT_SYMBOLS
)


def test_aami_mapping_complete():
    """Test all AAMI mappings are correct."""
    # Normal
    for sym in ["N", "L", "R", "e", "j"]:
        assert map_to_aami_class(sym) == "N"

    # Supraventricular
    for sym in ["A", "a", "J", "S"]:
        assert map_to_aami_class(sym) == "S"

    # Ventricular
    for sym in ["V", "E"]:
        assert map_to_aami_class(sym) == "V"

    # Fusion
    assert map_to_aami_class("F") == "F"

    # Excluded
    for sym in EXCLUDED_SYMBOLS:
        assert map_to_aami_class(sym) is None

    # Unknown symbols should return None (not default to N)
    for sym in ["Q", "/", "f", "Z", "X"]:
        assert map_to_aami_class(sym) is None


def test_filter_aami_beats_basic():
    r_peaks = np.array([100, 200, 300, 400, 500])
    labels = ["N", "V", "A", "F", "x"]
    filtered_peaks, filtered_labels, indices = filter_aami_beats(r_peaks, labels)
    assert len(filtered_peaks) == 4
    assert filtered_labels == ["N", "V", "S", "F"]
    assert np.array_equal(indices, [0, 1, 2, 3])


def test_segment_beats_boundary_handling():
    """Test segmentation handles boundaries correctly."""
    fs = 360
    # Signal where first and last beats are too close to boundaries
    signal = np.random.randn(2000)
    r_peaks = np.array([50, 500, 900, 1300, 1950])  # First and last near boundaries
    labels = ["N", "V", "N", "A", "F"]

    beats, beat_labels, indices = segment_beats_annotation_centered(
        signal, r_peaks, labels, fs, pre_window_s=0.25, post_window_s=0.45
    )
    # First beat (50) needs 90 samples before -> only 50 available, FAILS
    # Last beat (1950) needs 162 samples after -> only 50 available, FAILS
    # Middle 3 should work: indices 1, 2, 3 (0-indexed)
    assert len(beats) == 3
    assert beat_labels == ["V", "N", "S"]  # N, V, N, A, F -> V, N, S (middle 3)


def test_segment_beats_correct_window_size():
    """Test that window size is calculated from sampling frequency."""
    fs = 360
    signal = np.random.randn(10000)
    r_peaks = np.array([1000, 2000, 3000, 4000])
    labels = ["N", "V", "N", "A"]

    beats, beat_labels, indices = segment_beats_annotation_centered(
        signal, r_peaks, labels, fs, pre_window_s=0.25, post_window_s=0.45
    )

    expected_len = int(round(0.25 * fs)) + int(round(0.45 * fs)) + 1
    assert beats.shape[1] == expected_len
    # At 360 Hz: 90 + 162 + 1 = 253


def test_count_beats_per_class():
    labels = ["N", "L", "R", "e", "j",  # 5 Normal
              "A", "a", "J", "S",       # 4 Supraventricular
              "V", "E",                 # 2 Ventricular
              "F",                      # 1 Fusion
              "x", "!", "Q"]            # 3 Excluded
    counts = count_beats_per_class(labels)
    assert counts["N"] == 5
    assert counts["S"] == 4
    assert counts["V"] == 2
    assert counts["F"] == 1
    assert counts["excluded"] == 3


def test_get_aami_classes():
    classes = get_aami_classes()
    assert classes == ["N", "S", "V", "F"]


def test_aami_mapping_immutable():
    """Test that AAMI_MAPPING is not modified."""
    original = dict(AAMI_MAPPING)
    # Any operation
    _ = map_to_aami_class("N")
    assert AAMI_MAPPING == original


def test_unknown_symbol_is_not_silently_normal():
    """An unrecognised symbol must be excluded, never coerced to 'N'."""
    assert map_to_aami_class("Q") is None       # unclassifiable beat: not an AAMI class
    assert map_to_aami_class("/") is None       # paced beat: not an AAMI class
    assert map_to_aami_class("Z") is None       # unknown/garbage symbol
    assert map_to_aami_class("") is None
    # The genuinely unknown symbol must not be treated as a beat either
    assert is_beat_symbol("Z") is False


def test_is_beat_symbol_classification():
    """Beat detection is wider than AAMI classification but excludes artefact."""
    # Every AAMI-classifiable symbol is a beat
    for sym in AAMI_MAPPING:
        assert is_beat_symbol(sym) is True, sym

    # Beats excluded from the 4-class scheme still count as beats (they
    # delimit RR intervals) ...
    for sym in ["/", "f", "Q"]:
        assert is_beat_symbol(sym) is True, sym
        assert map_to_aami_class(sym) is None, sym

    # ... while rhythm / artefact / quality markers are not beats
    for sym in ["[", "]", "(", ")", "x", "!", "+", "~", "|"]:
        assert is_beat_symbol(sym) is False, sym


def test_segmentation_window_depends_on_fs():
    """Window length must be computed from fs, not a hardcoded 252 samples."""
    fs = 100  # slow recorder: 0.25 s -> 25 samples, 0.45 s -> 45 samples
    sig = np.arange(10000, dtype=float)
    r = np.array([500, 1500, 2500])
    labels = ["N", "N", "N"]

    beats, out_labels, idx = segment_beats_annotation_centered(sig, r, labels, fs)

    expected_len = int(round(0.25 * fs)) + int(round(0.45 * fs)) + 1  # 71
    assert beats.shape[1] == expected_len
    assert beats.shape[1] != 252


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])