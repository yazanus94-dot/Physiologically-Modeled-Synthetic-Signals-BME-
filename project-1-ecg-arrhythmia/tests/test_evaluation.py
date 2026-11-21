"""Tests for evaluation framework."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation import (
    get_classic_split, get_leakage_audited_split,
    verify_no_leakage, get_subject_for_record,
    split_records_by_subject, EvaluationSplit
)
from load_mitbih import MITBIH_RECORDS, SAME_SUBJECT_PAIRS


def test_classic_split_structure():
    """Test classic split has correct structure."""
    split = get_classic_split()
    assert isinstance(split, EvaluationSplit)
    assert split.protocol == "classic"
    assert len(split.train_records) == 18
    assert len(split.val_records) == 4
    assert len(split.test_records) == 22
    # No overlap between sets
    assert set(split.train_records).isdisjoint(split.val_records)
    assert set(split.train_records).isdisjoint(split.test_records)
    assert set(split.val_records).isdisjoint(split.test_records)


def test_classic_split_val_from_ds1():
    """Test validation records are subset of DS1."""
    split = get_classic_split()
    from load_mitbih import DS1_TRAIN_RECORDS
    for r in split.val_records:
        assert r in DS1_TRAIN_RECORDS
    for r in split.train_records:
        assert r in DS1_TRAIN_RECORDS


def test_leakage_audited_split_excludes_202():
    """Test leakage-audited split excludes record 202 from test."""
    split = get_leakage_audited_split()
    assert split.protocol == "leakage_audited"
    assert "201" in split.train_records
    assert "202" not in split.test_records
    assert len(split.test_records) == 21  # 22 - 1


def test_verify_no_leakage_classic_has_leakage():
    """Classic split has leakage (201 in train, 202 in test)."""
    split = get_classic_split()
    result = verify_no_leakage(split.train_records, split.test_records, verbose=False)
    assert result is False


def test_verify_no_leakage_audited_no_leakage():
    """Leakage-audited split has no leakage."""
    split = get_leakage_audited_split()
    result = verify_no_leakage(split.train_records, split.test_records, verbose=False)
    assert result is True


def test_get_subject_for_record():
    """Test subject mapping for records."""
    # Normal records
    assert get_subject_for_record("100") == "subject_100"
    assert get_subject_for_record("101") == "subject_101"

    # Same subject pair
    assert get_subject_for_record("201") == "subject_201_202"
    assert get_subject_for_record("202") == "subject_201_202"


def test_split_records_by_subject_groups_pairs():
    """Test random split keeps same-subject records together."""
    records = ["100", "101", "103", "201", "202", "203", "205"]
    split = split_records_by_subject(records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)

    # 201 and 202 must be in same set
    train_has_201 = "201" in split.train_records
    train_has_202 = "202" in split.train_records
    assert train_has_201 == train_has_202

    val_has_201 = "201" in split.val_records
    val_has_202 = "202" in split.val_records
    assert val_has_201 == val_has_202

    test_has_201 = "201" in split.test_records
    test_has_202 = "202" in split.test_records
    assert test_has_201 == test_has_202


def test_split_records_by_subject_no_leakage():
    """Test random subject split has no leakage."""
    records = ["100", "101", "103", "105", "106", "201", "202", "203", "205", "207"]
    split = split_records_by_subject(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    result = verify_no_leakage(split.train_records, split.test_records, verbose=False)
    assert result is True


def test_split_ratios_approximately_correct():
    """Test split ratios are approximately correct."""
    records = MITBIH_RECORDS[:20]  # Use subset
    split = split_records_by_subject(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    n_total = len(records)
    n_train = len(split.train_records)
    n_val = len(split.val_records)
    n_test = len(split.test_records)

    assert n_train + n_val + n_test == n_total
    # Ratios should be approximately correct (subject-level split may not be exact)
    assert 0.5 < n_train / n_total < 0.9
    assert 0.05 < n_val / n_total < 0.3
    assert 0.05 < n_test / n_total < 0.3


def test_evaluation_split_str():
    """Test EvaluationSplit string representation."""
    split = get_classic_split()
    s = str(split)
    assert "protocol=classic" in s
    assert "Train" in s
    assert "Val" in s
    assert "Test" in s


def test_classic_vs_audited_test_difference():
    """Test that audited test set is classic test minus 202."""
    classic = get_classic_split()
    audited = get_leakage_audited_split()

    classic_test = set(classic.test_records)
    audited_test = set(audited.test_records)

    assert audited_test == classic_test - {"202"}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])