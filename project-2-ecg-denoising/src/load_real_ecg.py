"""
Loads clean ECG recordings from the MIT-BIH Arrhythmia Database
(PhysioNet ``mitdb``) using the WFDB Python package.

Key points
----------
* Programmatic download into ``<repo>/data/physionet/mitdb/`` via
  ``wfdb.dl_database`` (that directory is git-ignored — raw PhysioNet
  data is never committed).
* The sampling frequency is always read from the WFDB record header
  (``record.fs``) — never hardcoded. MIT-BIH happens to be 360 Hz, but
  nothing in this module depends on that number.
* Missing or corrupted records are handled gracefully: the singular
  loaders raise precise exceptions, and the plural loaders
  (``load_ecg_records``) skip bad records with a printed warning so one
  bad file cannot kill a long experiment run.
* Reference R-peaks come from the record's ``.atr`` annotation file,
  restricted to beat annotations (see ``BEAT_ANNOTATION_SYMBOLS``).
"""
import os
import urllib.error
import urllib.request

import numpy as np
import wfdb

PHYSIONET_FILES_URL = "https://physionet.org/files"
DATABASE_VERSIONS = {"mitdb": "1.0.0", "nstdb": "1.0.0"}
HTTP_TIMEOUT_S = 60
USER_AGENT = ("Mozilla/5.0 (compatible; ecg-denoising-research-scripts/1.0; "
              "+https://physionet.org)")

# Repository root: .../Physiologically-Modeled-Synthetic-Signals-BME-
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         os.pardir, os.pardir))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "physionet")
MITDB_DIR = os.path.join(DATA_ROOT, "mitdb")

# WFDB annotation symbols that denote a ventricular/supraventricular
# *beat* (QRS). Rhythm markers ('+', '"'), wave markers ('p', 't'),
# signal-quality markers ('~') and non-beat artefacts ('x', '[') are
# excluded. 'r' is the R-on-T PVC, which is still a beat.
BEAT_ANNOTATION_SYMBOLS = frozenset({
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "F", "Q",
    "e", "j", "n", "E", "/", "f", "r",
})


def fetch_text(url, timeout=HTTP_TIMEOUT_S):
    """Download a small text file from PhysioNet (used to read RECORDS /
    documentation files programmatically instead of hardcoding them)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def list_remote_records(db, version=None):
    """Return the official record list of a PhysioNet database by fetching
    its ``RECORDS`` file (no guessing — this is what PhysioNet serves)."""
    version = version or DATABASE_VERSIONS.get(db, "1.0.0")
    url = f"{PHYSIONET_FILES_URL}/{db}/{version}/RECORDS"
    text = fetch_text(url, timeout=HTTP_TIMEOUT_S)
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def records_dir(db):
    return os.path.join(DATA_ROOT, db)


def local_records(db):
    """Record names that already exist locally (scan ``*.hea`` headers).
    Returns [] without touching the network, so this is safe offline."""
    directory = records_dir(db)
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(directory)
                  if f.endswith(".hea"))


def is_downloaded(db, records=None):
    """True when every requested record has both a header and a signal
    file on disk. With ``records=None``: True when the database directory
    contains at least one record."""
    directory = records_dir(db)
    names = records if records is not None else local_records(db)
    if records is None:
        return len(names) > 0
    for name in names:
        if not (os.path.isfile(os.path.join(directory, name + ".hea")) and
                os.path.isfile(os.path.join(directory, name + ".dat"))):
            return False
    return len(names) > 0


def download_records(db, records="all", dl_dir=None, annotators=None):
    """Download records from PhysioNet into ``data/physionet/<db>/``.

    ``annotators=None`` skips annotation files (used for nstdb, whose pure
    noise records have none); pass e.g. ``["atr"]`` for mitdb.
    """
    dl_dir = dl_dir or records_dir(db)
    os.makedirs(dl_dir, exist_ok=True)
    wfdb.dl_database(db, dl_dir, records=records, annotators=annotators)
    return dl_dir


def _read_beat_annotations(path, start, end):
    """Beat annotation samples inside [start, end), shifted so that 0 is
    the first sample of the returned segment. Returns None if the
    annotation file is absent."""
    if not os.path.isfile(path + ".atr"):
        return None
    ann = wfdb.rdann(path, "atr")
    samples = np.asarray(ann.sample, dtype=int)
    symbols = list(ann.symbol)
    beat_mask = np.array([s in BEAT_ANNOTATION_SYMBOLS for s in symbols],
                         dtype=bool)
    selected = samples[beat_mask]
    selected = selected[(selected >= start) & (selected < end)]
    return selected - start


def _clean_invalid_samples(signal, record_name, sig_name):
    """WFDB physical signals contain NaN for invalid samples — replace
    them by linear interpolation so downstream filters don't return an
    all-NaN array (a corrupted-sample warning is printed)."""
    bad = ~np.isfinite(signal)
    if bad.all():
        raise RuntimeError(f"record {record_name!r} channel {sig_name!r}: "
                           f"all samples are invalid")
    if bad.any():
        idx = np.arange(signal.size)
        signal = signal.copy()
        signal[bad] = np.interp(idx[bad], idx[~bad], signal[~bad])
        print(f"  [warn] record {record_name!r}: replaced "
              f"{int(bad.sum())} invalid samples in channel {sig_name!r}")
    return signal


def load_ecg_record(name, channel=0, db="mitdb", offset_s=0.0,
                    duration_s=None):
    """Load one MIT-BIH record (or raise a precise exception).

    Returns a dict with keys: name, signal, fs, units, sig_name,
    r_peaks (np.ndarray of reference beat positions in the returned
    segment, samples), offset_s, duration_s.
    """
    directory = records_dir(db)
    path = os.path.join(directory, name)
    if not os.path.isfile(path + ".hea"):
        raise FileNotFoundError(
            f"record {name!r} not found in {directory} — download it first "
            f"(e.g. run run_real_experiment.py --download, or call "
            f"download_records('{db}', records=['{name}']))")
    try:
        record = wfdb.rdrecord(path)
    except Exception as exc:  # noqa: BLE001 - corrupted/unsupported file
        raise RuntimeError(f"corrupted record {name!r} in {directory}: "
                           f"{exc}") from exc

    fs = float(record.fs)  # read from the WFDB header — never hardcoded
    if channel >= record.n_sig:
        raise ValueError(f"record {name!r} has {record.n_sig} channel(s); "
                         f"channel {channel} is out of range")

    start = int(round(offset_s * fs))
    if start < 0:
        raise ValueError(f"offset_s must be >= 0, got {offset_s}")
    stop = record.sig_len if duration_s is None else \
        start + int(round(duration_s * fs))
    stop = min(stop, record.sig_len)
    if stop <= start:
        raise ValueError(f"record {name!r} ({record.sig_len} samples at "
                         f"{fs:g} Hz) has no samples in "
                         f"[{offset_s}s, {offset_s + (duration_s or 0):g}s]")

    signal = np.asarray(record.p_signal[start:stop, channel], dtype=float)
    sig_name = str(record.sig_name[channel])
    signal = _clean_invalid_samples(signal, name, sig_name)

    r_peaks = _read_beat_annotations(path, start, stop)

    return {
        "name": name,
        "signal": signal,
        "fs": fs,
        "units": str(record.units[channel]),
        "sig_name": sig_name,
        "r_peaks": r_peaks,
        "offset_s": offset_s,
        "duration_s": (stop - start) / fs,
    }


def load_ecg_records(names, **kwargs):
    """Load several records, skipping (with a printed message) any that
    are missing or corrupted instead of aborting the whole run."""
    loaded = []
    for name in names:
        try:
            loaded.append(load_ecg_record(name, **kwargs))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"  [skip] ECG record {name!r}: {exc}")
    return loaded
