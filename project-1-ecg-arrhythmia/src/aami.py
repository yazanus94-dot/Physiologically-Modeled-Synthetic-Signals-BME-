"""
AAMI EC57 beat classification mapping and beat segmentation.

Implements the standard AAMI 4-class beat classification scheme
from the MIT-BIH Arrhythmia Database annotations.

AAMI Classes:
- N (Normal): N, L, R, e, j
- S (Supraventricular ectopic): A, a, J, S
- V (Ventricular ectopic): V, E
- F (Fusion): F
"""
from typing import Dict, List, Optional, Tuple
import numpy as np


# AAMI EC57 standard annotation mapping
AAMI_MAPPING = {
    # Normal beats
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat

    # Supraventricular ectopic beats
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Supraventricular premature beat

    # Ventricular ectopic beats
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat

    # Fusion beats
    "F": "F",  # Fusion of ventricular and normal beat
}

# Symbols to exclude from classification (non-beat or unsupported)
EXCLUDED_SYMBOLS = {
    "[", "]",  # Rhythm annotations
    "(", ")",  # Rhythm annotations
    "x",       # Non-conducted P-wave (not a beat)
    "!",       # Ventricular flutter wave
    "?",       # Beat not classified
    "+",       # Rhythm change
    "~",       # Signal quality change
    "|",       # Isolated QRS-like artifact
}

# Symbols that denote an actual ventricular depolarisation (a "beat") even when
# they are NOT part of the four AAMI outcome classes. RR intervals must be
# measured between consecutive *beats* — e.g. a paced beat ("/") or an
# unclassifiable beat ("Q") still separates two RR intervals in time, so
# dropping it would silently merge two real intervals.
BEAT_SYMBOLS = {
    # The 14 AAMI-classifiable beat symbols
    "N", "L", "R", "e", "j",
    "A", "a", "J", "S",
    "V", "E",
    "F",
    # Beats that are valid QRS events but excluded from the 4-class scheme
    "/",   # Paced beat
    "f",   # Fusion of paced and normal beat
    "Q",   # Unclassifiable beat
}

# Symbols that are definitely NOT beats (artefact / rhythm / quality markers)
NON_BEAT_SYMBOLS = EXCLUDED_SYMBOLS

# All valid MIT-BIH symbols that map to AAMI classes
VALID_AAMI_SYMBOLS = set(AAMI_MAPPING.keys())

# All known MIT-BIH symbols (for validation)
ALL_MITBIH_SYMBOLS = {
    "N", "L", "R", "e", "j",  # Normal
    "A", "a", "J", "S",        # Supraventricular
    "V", "E",                  # Ventricular
    "F",                       # Fusion
    "/", "f", "Q",             # Other (excluded from AAMI)
    "[", "]", "(", ")", "x", "!", "?", "+", "~", "|",  # Non-beat
}


def map_to_aami_class(symbol: str) -> Optional[str]:
    """
    Map MIT-BIH annotation symbol to AAMI class.

    Parameters
    ----------
    symbol : str
        MIT-BIH annotation symbol

    Returns
    -------
    str or None
        AAMI class ('N', 'S', 'V', 'F') or None if symbol
        should be excluded from classification
    """
    if symbol in EXCLUDED_SYMBOLS:
        return None
    return AAMI_MAPPING.get(symbol, None)


def is_beat_symbol(symbol: str) -> bool:
    """
    True when the annotation marks a real QRS complex (a heartbeat).

    This is deliberately *wider* than ``map_to_aami_class``: paced ('/'),
    fusion-paced ('f') and unclassifiable ('Q') beats are not members of the
    four AAMI outcome classes, but they are still beats and therefore still
    delimit RR intervals.
    """
    return symbol in BEAT_SYMBOLS


def get_aami_classes() -> List[str]:
    """Return list of AAMI class labels in standard order."""
    return ["N", "S", "V", "F"]


def get_symbol_to_aami_mapping() -> Dict[str, str]:
    """Return the full symbol-to-AAMI mapping dictionary."""
    return AAMI_MAPPING.copy()


def count_beats_per_class(labels: List[str]) -> Dict[str, int]:
    """Count beats per AAMI class from annotation symbols."""
    counts = {cls: 0 for cls in get_aami_classes()}
    excluded = 0

    for sym in labels:
        aami = map_to_aami_class(sym)
        if aami is None:
            excluded += 1
        else:
            counts[aami] += 1

    counts["excluded"] = excluded
    return counts


def filter_aami_beats(
    r_peaks: np.ndarray,
    labels: List[str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Filter beats to only include those with valid AAMI classes.

    Parameters
    ----------
    r_peaks : np.ndarray
        Sample indices of beat annotations
    labels : list of str
        Annotation symbols

    Returns
    -------
    filtered_peaks : np.ndarray
        R-peaks with valid AAMI classes
    filtered_labels : list of str
        Corresponding AAMI class labels
    original_indices : np.ndarray
        Indices into original arrays
    """
    valid_mask = []
    aami_labels = []

    for i, sym in enumerate(labels):
        aami = map_to_aami_class(sym)
        if aami is not None:
            valid_mask.append(i)
            aami_labels.append(aami)

    return r_peaks[valid_mask], aami_labels, np.array(valid_mask)


def segment_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    pre_window_s: float = 0.25,
    post_window_s: float = 0.45
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment ECG signal into individual beats centered on R-peaks.

    Parameters
    ----------
    signal : np.ndarray
        Full ECG signal
    r_peaks : np.ndarray
        Sample indices of R-peaks (beat centers)
    fs : int
        Sampling frequency in Hz
    pre_window_s : float
        Seconds before R-peak to include (default 0.25s)
    post_window_s : float
        Seconds after R-peak to include (default 0.45s)

    Returns
    -------
    beats : np.ndarray
        Shape (n_beats, window_samples) - segmented beats
    valid_indices : np.ndarray
        Indices of beats that were successfully segmented (not truncated)
    """
    pre_samples = int(round(pre_window_s * fs))
    post_samples = int(round(post_window_s * fs))
    window_len = pre_samples + post_samples + 1

    beats = []
    valid_indices = []

    for i, r in enumerate(r_peaks):
        start = r - pre_samples
        end = r + post_samples + 1

        # Check boundaries
        if start < 0 or end > len(signal):
            continue  # Skip beats at boundaries

        beat = signal[start:end]
        if len(beat) == window_len:
            beats.append(beat)
            valid_indices.append(i)

    if not beats:
        return np.empty((0, window_len)), np.array([], dtype=int)

    return np.array(beats), np.array(valid_indices)


def segment_beats_annotation_centered(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    labels: List[str],
    fs: int,
    pre_window_s: float = 0.25,
    post_window_s: float = 0.45
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Segment beats using annotation positions (not detected peaks).

    This is for annotation-centered classification, separate from
    R-peak detector evaluation.

    Parameters
    ----------
    signal : np.ndarray
        Full ECG signal
    r_peaks : np.ndarray
        Annotation sample positions (reference R-peaks)
    labels : list of str
        Annotation symbols
    fs : int
        Sampling frequency
    pre_window_s : float
        Seconds before annotation
    post_window_s : float
        Seconds after annotation

    Returns
    -------
    beats : np.ndarray
        Segmented beats (n_beats, window_samples)
    aami_labels : list of str
        AAMI class for each beat
    valid_indices : np.ndarray
        Indices into original annotation arrays
    """
    # First filter to valid AAMI beats
    filtered_peaks, aami_labels, orig_indices = filter_aami_beats(r_peaks, labels)

    # Then segment
    beats, valid_rel_indices = segment_beats(
        signal, filtered_peaks, fs, pre_window_s, post_window_s
    )

    # Map back to original indices
    final_indices = orig_indices[valid_rel_indices]
    final_labels = [aami_labels[i] for i in valid_rel_indices]

    return beats, final_labels, final_indices


if __name__ == "__main__":
    # Quick test
    test_symbols = ["N", "V", "A", "F", "L", "R", "E", "j", "S", "a", "J", "Q", "/", "x", "!"]
    print("Symbol -> AAMI mapping:")
    for sym in test_symbols:
        aami = map_to_aami_class(sym)
        print(f"  {sym:2s} -> {aami}")

    print("\nBeat counts test:")
    labels = ["N", "N", "V", "A", "F", "L", "R", "x", "!", "Q"]
    print(count_beats_per_class(labels))

    print("\nSegmentation test:")
    fs = 360
    signal = np.sin(np.arange(3600) * 2 * np.pi * 1.0 / fs)  # 10 sec, 1 Hz sine
    r_peaks = np.array([360, 720, 1080, 1440, 1800])  # Every 1 second
    labels = ["N", "V", "N", "A", "F"]
    beats, beat_labels, indices = segment_beats_annotation_centered(signal, r_peaks, labels, fs)
    print(f"Segmented {len(beats)} beats, shape: {beats.shape}")
    print(f"Labels: {beat_labels}")
    print(f"Original indices: {indices}")