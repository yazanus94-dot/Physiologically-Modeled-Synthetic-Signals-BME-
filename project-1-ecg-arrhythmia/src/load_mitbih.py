"""
MIT-BIH Arrhythmia Database loader.

Downloads, caches, and loads real PhysioNet MIT-BIH ECG recordings
using the `wfdb` package. Provides clean interface for ECG waveform,
annotations, and metadata extraction.
"""
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# Optional dependency - only needed for real data loading
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    wfdb = None


# MIT-BIH record names (48 records total).
# 102, 104, 107 and 217 are part of the database but are NOT assigned to
# either DS1 or DS2 in the standard benchmark split.
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

# Records from the same subject (for leakage audit)
SAME_SUBJECT_PAIRS = {
    "201": "202",  # 201 and 202 are from the same patient
    "202": "201",
}

# DS1/DS2 splits (standard published benchmark)
DS1_TRAIN_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230"
]

DS2_TEST_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234"
]

# Preferred ECG leads in order of preference
PREFERRED_LEADS = ["MLII", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


class MITBIHLoader:
    """Load and manage MIT-BIH Arrhythmia Database recordings."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize loader.

        Parameters
        ----------
        data_dir : str, optional
            Directory to store downloaded data. Defaults to ./data/physionet/mitdb/
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "physionet", "mitdb")
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not WFDB_AVAILABLE:
            warnings.warn(
                "wfdb package not installed. Install with: pip install wfdb\n"
                "Real MIT-BIH data loading will not work.",
                ImportWarning
            )

    def download_database(self, records: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Download MIT-BIH database records.

        Parameters
        ----------
        records : list of str, optional
            Specific records to download. If None, downloads all 48 records.

        Returns
        -------
        dict
            Mapping of record name to download success (True/False)
        """
        if not WFDB_AVAILABLE:
            raise RuntimeError("wfdb package required for downloading. Install with: pip install wfdb")

        if records is None:
            records = MITBIH_RECORDS

        results = {}
        for record in records:
            try:
                wfdb.dl_database("mitdb", str(self.data_dir), records=[record])
                results[record] = True
                print(f"Downloaded record {record}")
            except Exception as e:
                print(f"Failed to download record {record}: {e}")
                results[record] = False
        return results

    def is_record_available(self, record: str) -> bool:
        """Check if a record exists locally."""
        record_path = self.data_dir / record
        return record_path.with_suffix(".dat").exists() and record_path.with_suffix(".hea").exists()

    def get_available_records(self) -> List[str]:
        """Get list of locally available records."""
        return [r for r in MITBIH_RECORDS if self.is_record_available(r)]

    def _select_ecg_channel(self, record: wfdb.Record) -> Tuple[np.ndarray, str]:
        """
        Select the best available ECG channel.

        Prefers MLII (lead II) as it's the standard analysis lead.
        Falls back to other leads if unavailable.

        Returns
        -------
        signal : np.ndarray
            Selected ECG channel signal
        channel_name : str
            Name of selected channel
        """
        sig_names = [name.upper() for name in record.sig_name]

        for preferred in PREFERRED_LEADS:
            if preferred.upper() in sig_names:
                idx = sig_names.index(preferred.upper())
                return record.p_signal[:, idx], record.sig_name[idx]

        # Fallback: use first channel
        warnings.warn(
            f"No preferred lead found in record. Available: {record.sig_name}. "
            f"Using channel 0 ({record.sig_name[0]}).",
            UserWarning
        )
        return record.p_signal[:, 0], record.sig_name[0]

    def load_record(
        self,
        record_name: str,
        annotation_ext: str = "atr",
        channel: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a single MIT-BIH record with annotations.

        Parameters
        ----------
        record_name : str
            Record identifier (e.g., "100", "101", etc.)
        annotation_ext : str
            Annotation extension (default "atr" for beat annotations)
        channel : str, optional
            Specific channel to load. If None, auto-selects preferred lead.

        Returns
        -------
        dict with keys:
            - signal: np.ndarray, ECG waveform
            - fs: int, sampling frequency
            - r_peaks: np.ndarray, sample indices of beat annotations
            - labels: list of str, annotation symbols
            - channel_name: str, name of loaded channel
            - record_name: str
            - n_samples: int
            - duration_s: float
        """
        if not WFDB_AVAILABLE:
            raise RuntimeError("wfdb package required. Install with: pip install wfdb")

        if not self.is_record_available(record_name):
            raise FileNotFoundError(
                f"Record {record_name} not found in {self.data_dir}. "
                f"Run download_database() first."
            )

        record_path = str(self.data_dir / record_name)

        # Load record
        record = wfdb.rdrecord(record_path)
        fs = record.fs

        # Select channel
        if channel and channel in record.sig_name:
            idx = record.sig_name.index(channel)
            signal = record.p_signal[:, idx]
            channel_name = channel
        else:
            signal, channel_name = self._select_ecg_channel(record)

        # Load annotations
        ann = wfdb.rdann(record_path, annotation_ext)
        r_peaks = ann.sample
        labels = ann.symbol

        return {
            "signal": signal,
            "fs": fs,
            "r_peaks": r_peaks,
            "labels": labels,
            "channel_name": channel_name,
            "record_name": record_name,
            "n_samples": len(signal),
            "duration_s": len(signal) / fs
        }

    def load_multiple_records(
        self,
        record_names: List[str],
        annotation_ext: str = "atr"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load multiple records.

        Returns
        -------
        dict mapping record_name to record data (same keys as load_record)
        """
        results = {}
        for record_name in record_names:
            try:
                results[record_name] = self.load_record(record_name, annotation_ext)
            except Exception as e:
                warnings.warn(f"Failed to load record {record_name}: {e}")
        return results

    def get_record_info(self, record_name: str) -> Dict[str, Any]:
        """Get metadata for a record without loading full signal."""
        if not self.is_record_available(record_name):
            raise FileNotFoundError(f"Record {record_name} not found")

        record_path = str(self.data_dir / record_name)
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, "atr")

        return {
            "record_name": record_name,
            "fs": record.fs,
            "n_samples": int(record.sig_len),
            "channels": record.sig_name,
            "n_annotations": len(ann.sample),
            "annotation_symbols": list(set(ann.symbol)),
            "duration_s": record.sig_len / record.fs
        }


def get_default_data_dir() -> Path:
    """Get the default data directory path."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "physionet" / "mitdb"


if __name__ == "__main__":
    # Quick test
    loader = MITBIHLoader()
    print(f"Data directory: {loader.data_dir}")
    print(f"Available records: {loader.get_available_records()}")

    if loader.get_available_records():
        rec = loader.load_record(loader.get_available_records()[0])
        print(f"Loaded {rec['record_name']}: fs={rec['fs']}Hz, "
              f"{len(rec['signal'])} samples, {len(rec['r_peaks'])} beats")