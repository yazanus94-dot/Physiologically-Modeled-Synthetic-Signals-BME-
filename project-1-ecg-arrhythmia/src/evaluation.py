"""
Patient-independent evaluation framework for MIT-BIH arrhythmia classification.

Implements:
- Classic DS1/DS2 record split (published benchmark)
- Leakage-audited split (excludes same-subject records across train/test)
- Validation split within DS1 using record grouping
- No random shuffling of individual beats
- Proper subject-level separation
"""
import warnings
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np

# Import our splits from load_mitbih
from load_mitbih import (
    DS1_TRAIN_RECORDS,
    DS2_TEST_RECORDS,
    SAME_SUBJECT_PAIRS,
    MITBIH_RECORDS
)

# Canonical feature vector for the MIT-BIH experiment:
# 5 RR-interval features + 9 beat-window morphology features = 14 columns.
EXPECTED_FEATURE_NAMES = [
    "rr_prev", "rr_next", "rr_mean_local", "rr_ratio", "prematurity_ratio",
    "qrs_amp", "qrs_width", "qrs_energy",
    "beat_mean", "beat_std", "beat_var",
    "beat_skew", "beat_kurtosis",
    "r_amp_rel",
]


@dataclass
class EvaluationSplit:
    """Container for train/validation/test record splits."""
    train_records: List[str]
    val_records: List[str]
    test_records: List[str]
    protocol: str  # "classic" or "leakage_audited"
    description: str

    def __str__(self):
        return (
            f"EvaluationSplit(protocol={self.protocol})\n"
            f"  Train ({len(self.train_records)}): {self.train_records}\n"
            f"  Val ({len(self.val_records)}): {self.val_records}\n"
            f"  Test ({len(self.test_records)}): {self.test_records}"
        )


def get_classic_split() -> EvaluationSplit:
    """
    Get the classic DS1/DS2 split as published in literature.

    Note: Records 201 and 202 are from the same subject, which creates
    potential data leakage between DS1 (201 is in the training pool) and
    DS2 (202 is in the test set).  Classic is the "as published"
    benchmark; see ``get_leakage_audited_split`` for the variant where
    that leakage is guaranteed removed.
    """
    # Use a subset of DS1 for validation (hold out ~20%)
    # Group by subject to avoid leakage within DS1
    val_records = ["106", "119", "203", "220"]  # ~20% of DS1, diverse subjects
    train_records = [r for r in DS1_TRAIN_RECORDS if r not in val_records]

    return EvaluationSplit(
        train_records=train_records,
        val_records=val_records,
        test_records=DS2_TEST_RECORDS,
        protocol="classic",
        description="Classic DS1/DS2 split with validation subset from DS1"
    )


def get_leakage_audited_split() -> EvaluationSplit:
    """
    Get the leakage-audited split.

    Ensures no subject appears in both training and testing.  Since
    201 and 202 are from the same subject:
    - Keep 201 in training (it's in DS1)
    - Exclude 202 from test set (it's in DS2)

    The exclusion is *audited at runtime*: the split is built from the
    classic protocol and ``verify_no_leakage`` is called on it, raising
    if any subject is shared between train and test rather than assuming
    the exclusion is sufficient.
    """
    # Start with classic split
    classic = get_classic_split()

    # Remove 202 from test set
    test_records = [r for r in classic.test_records if r != "202"]

    audited = EvaluationSplit(
        train_records=classic.train_records,
        val_records=classic.val_records,
        test_records=test_records,
        protocol="leakage_audited",
        description=("Leakage-audited: 202 excluded from test (same "
                     "subject as 201 in train); "
                     "verify_no_leakage asserts no shared subject between "
                     "train and test"),
    )
    ok = verify_no_leakage(audited.train_records, audited.test_records)
    if not ok:
        raise RuntimeError(
            "leakage audit failed for leakage_audited split: a subject "
            "appears in both train and test")
    return audited


def get_subject_for_record(record: str) -> str:
    """
    Map record to subject identifier.

    For MIT-BIH, most records are unique subjects except 201/202.
    """
    if record in SAME_SUBJECT_PAIRS:
        # Use sorted order for canonical subject ID
        pair = tuple(sorted([record, SAME_SUBJECT_PAIRS[record]]))
        return f"subject_{pair[0]}_{pair[1]}"
    return f"subject_{record}"


def verify_no_leakage(
    train_records: List[str],
    test_records: List[str],
    verbose: bool = True
) -> bool:
    """
    Verify no subject appears in both train and test sets.

    Returns True if no leakage, False otherwise.
    """
    train_subjects = {get_subject_for_record(r) for r in train_records}
    test_subjects = {get_subject_for_record(r) for r in test_records}

    overlap = train_subjects & test_subjects

    if verbose and overlap:
        warnings.warn(f"LEAKAGE DETECTED: Shared subjects: {overlap}")
    elif verbose:
        print("No subject leakage detected.")

    return len(overlap) == 0


def get_record_to_subject_map(records: List[str]) -> Dict[str, str]:
    """Map each record to its subject."""
    return {r: get_subject_for_record(r) for r in records}


def split_records_by_subject(
    records: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> EvaluationSplit:
    """
    Split records by subject (not by individual record) to prevent leakage.

    This is an alternative splitting strategy for custom experiments.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    # Group records by subject
    subject_to_records = {}
    for r in records:
        subj = get_subject_for_record(r)
        if subj not in subject_to_records:
            subject_to_records[subj] = []
        subject_to_records[subj].append(r)

    subjects = list(subject_to_records.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    train_records = [r for s in train_subjects for r in subject_to_records[s]]
    val_records = [r for s in val_subjects for r in subject_to_records[s]]
    test_records = [r for s in test_subjects for r in subject_to_records[s]]

    return EvaluationSplit(
        train_records=sorted(train_records),
        val_records=sorted(val_records),
        test_records=sorted(test_records),
        protocol="subject_random",
        description=f"Random subject-level split (train={train_ratio}, val={val_ratio}, test={test_ratio})"
    )


def collect_beats_from_records(
    loader,
    record_names: List[str],
    fs_target: Optional[int] = None,
    preprocess_config=None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Collect all beats from a list of records.

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        AAMI class labels
    record_ids : list of str
        Record name for each beat
    beat_indices : list of int
        Beat index within record
    """
    from aami import segment_beats_annotation_centered, filter_aami_beats
    from features import extract_features_from_beats, extract_rr_features_for_annotations
    from preprocessing import preprocess_ecg, PreprocessingConfig

    if preprocess_config is None:
        preprocess_config = PreprocessingConfig()

    all_beats = []
    all_labels = []
    all_record_ids = []
    all_beat_indices = []
    feat_names_out = None

    for record_name in record_names:
        try:
            rec_data = loader.load_record(record_name)
        except Exception as e:
            warnings.warn(f"Failed to load {record_name}: {e}")
            continue

        signal = rec_data["signal"]
        fs = rec_data["fs"]
        r_peaks = rec_data["r_peaks"]
        labels = rec_data["labels"]

        # Resample if needed
        if fs_target is not None and fs != fs_target:
            from scipy.signal import resample
            n_new = int(len(signal) * fs_target / fs)
            signal = resample(signal, n_new)
            r_peaks = (r_peaks * fs_target / fs).astype(int)
            fs = fs_target

        # Preprocess
        signal, _ = preprocess_ecg(signal, fs, preprocess_config)

        # Segment beats around annotations (annotation-centered)
        beats, aami_labels, orig_indices = segment_beats_annotation_centered(
            signal, r_peaks, labels, fs
        )

        if len(beats) == 0:
            continue

        # RR features rebuilt from the full annotation series so that the
        # prematurity context is not lost when non-AAMI beats are excluded.
        rr_feat, rr_names, keep = extract_rr_features_for_annotations(
            r_peaks, labels, orig_indices, fs
        )
        if rr_feat.shape[0] == 0:
            warnings.warn(f"{record_name}: no beat with both RR neighbours - skipped")
            continue
        if keep.size != len(beats):  # defensive: shapes must correspond
            warnings.warn(
                f"{record_name}: RR/beat alignment mismatch "
                f"({keep.size} vs {len(beats)}) - record skipped"
            )
            continue

        beats = beats[keep]
        aami_labels = [lab for lab, k in zip(aami_labels, keep) if k]
        orig_indices = orig_indices[keep]

        morph, morph_names = extract_features_from_beats(beats, np.array([]), fs)
        if morph.shape[0] != rr_feat.shape[0]:  # defensive
            warnings.warn(f"{record_name}: feature row mismatch - record skipped")
            continue

        X_rec = np.hstack([rr_feat, morph])
        if feat_names_out is None:
            feat_names_out = list(rr_names) + list(morph_names)

        all_beats.append(X_rec)
        all_labels.extend(aami_labels)
        all_record_ids.extend([record_name] * X_rec.shape[0])
        all_beat_indices.extend(orig_indices.tolist())

    if not all_beats or feat_names_out is None:
        return np.empty((0, 14)), np.array([]), [], [], EXPECTED_FEATURE_NAMES

    X = np.vstack(all_beats)
    return X, np.array(all_labels), all_record_ids, all_beat_indices, feat_names_out


if __name__ == "__main__":
    # Test splits
    print("=" * 60)
    print("CLASSIC SPLIT")
    print("=" * 60)
    classic = get_classic_split()
    print(classic)
    print()
    verify_no_leakage(classic.train_records, classic.test_records)

    print("\n" + "=" * 60)
    print("LEAKAGE-AUDITED SPLIT")
    print("=" * 60)
    audited = get_leakage_audited_split()
    print(audited)
    print()
    verify_no_leakage(audited.train_records, audited.test_records)

    print("\n" + "=" * 60)
    print("RANDOM SUBJECT SPLIT")
    print("=" * 60)
    random_split = split_records_by_subject(MITBIH_RECORDS)
    print(random_split)
    verify_no_leakage(random_split.train_records, random_split.test_records)