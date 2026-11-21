"""Tests for MIT-BIH loader (without requiring actual data)."""
import sys
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from load_mitbih import (
    MITBIHLoader, MITBIH_RECORDS, DS1_TRAIN_RECORDS, DS2_TEST_RECORDS,
    SAME_SUBJECT_PAIRS, PREFERRED_LEADS
)


def test_mitbih_records_complete():
    """The record list must match the full 48-record MIT-BIH database."""
    # All 48 records exist in the database. 102, 104, 107 and 217 are real,
    # valid recordings - they are simply not assigned to DS1 or DS2 by the
    # standard benchmark split (that is handled by the DS1/DS2 constants).
    assert len(MITBIH_RECORDS) == 48
    assert len(set(MITBIH_RECORDS)) == 48

    # Check some known records, including the four outside DS1/DS2
    for rec in ["100", "101", "234", "102", "104", "107", "217"]:
        assert rec in MITBIH_RECORDS

    # Every DS1/DS2 record must be a known MIT-BIH record
    assert set(DS1_TRAIN_RECORDS).issubset(set(MITBIH_RECORDS))
    assert set(DS2_TEST_RECORDS).issubset(set(MITBIH_RECORDS))


def test_ds1_ds2_splits():
    """Test DS1/DS2 splits are correct."""
    assert len(DS1_TRAIN_RECORDS) == 22
    assert len(DS2_TEST_RECORDS) == 22
    # No overlap
    assert set(DS1_TRAIN_RECORDS).isdisjoint(DS2_TEST_RECORDS)
    # All 44 records accounted for
    assert len(set(DS1_TRAIN_RECORDS) | set(DS2_TEST_RECORDS)) == 44
    # 4 records not in either (102, 104, 107, 217 - known problematic records)


def test_same_subject_pairs():
    """Test same subject mapping."""
    assert SAME_SUBJECT_PAIRS["201"] == "202"
    assert SAME_SUBJECT_PAIRS["202"] == "201"
    assert len(SAME_SUBJECT_PAIRS) == 2


def test_preferred_leads():
    """Test preferred leads ordering."""
    assert PREFERRED_LEADS[0] == "MLII"
    assert "II" in PREFERRED_LEADS
    assert "V1" in PREFERRED_LEADS


def test_loader_initialization():
    """Test loader initialization creates data directory."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = MITBIHLoader(data_dir=tmpdir)
        assert loader.data_dir.exists()
        assert loader.data_dir == Path(tmpdir).resolve()


def test_is_record_available_false():
    """Test record availability check returns False for missing records."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = MITBIHLoader(data_dir=tmpdir)
        assert not loader.is_record_available("100")


def test_get_available_records_empty():
    """Test getting available records when none downloaded."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = MITBIHLoader(data_dir=tmpdir)
        assert loader.get_available_records() == []


@patch("load_mitbih.wfdb")
def test_download_database(mock_wfdb):
    """Test database download calls wfdb correctly."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = MITBIHLoader(data_dir=tmpdir)
        loader.download_database(["100", "101"])
        assert mock_wfdb.dl_database.call_count == 2


@patch("load_mitbih.wfdb")
def test_load_record_channel_selection(mock_wfdb):
    """Test automatic channel selection prefers MLII."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock wfdb record
        mock_record = MagicMock()
        mock_record.fs = 360
        mock_record.sig_name = ["MLII", "V1"]
        mock_record.p_signal = np.random.randn(3600, 2)
        mock_record.n_sig = 2

        mock_ann = MagicMock()
        mock_ann.sample = np.array([360, 720, 1080])
        mock_ann.symbol = np.array(["N", "N", "V"])

        mock_wfdb.rdrecord.return_value = mock_record
        mock_wfdb.rdann.return_value = mock_ann

        loader = MITBIHLoader(data_dir=tmpdir)
        # Create dummy files so is_record_available returns True
        (Path(tmpdir) / "100.dat").touch()
        (Path(tmpdir) / "100.hea").touch()

        result = loader.load_record("100")

        assert result["fs"] == 360
        assert result["channel_name"] == "MLII"
        assert len(result["signal"]) == 3600
        assert len(result["r_peaks"]) == 3


@patch("load_mitbih.wfdb")
def test_load_record_fallback_channel(mock_wfdb):
    """Test fallback to first channel when preferred not available."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_record = MagicMock()
        mock_record.fs = 360
        mock_record.sig_name = ["V1", "V2"]  # No MLII or II
        mock_record.p_signal = np.random.randn(3600, 2)
        mock_record.n_sig = 2

        mock_ann = MagicMock()
        mock_ann.sample = np.array([360, 720])
        mock_ann.symbol = np.array(["N", "N"])

        mock_wfdb.rdrecord.return_value = mock_record
        mock_wfdb.rdann.return_value = mock_ann

        loader = MITBIHLoader(data_dir=tmpdir)
        (Path(tmpdir) / "100.dat").touch()
        (Path(tmpdir) / "100.hea").touch()

        result = loader.load_record("100")
        assert result["channel_name"] == "V1"  # Falls back to first


def test_get_default_data_dir():
    """Test default data directory path."""
    from load_mitbih import get_default_data_dir
    path = get_default_data_dir()
    assert "data" in str(path)
    assert "physionet" in str(path)
    assert "mitdb" in str(path)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])