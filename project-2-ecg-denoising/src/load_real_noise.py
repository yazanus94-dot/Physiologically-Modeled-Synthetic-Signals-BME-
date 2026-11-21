"""
Loads real noise recordings from the MIT-BIH Noise Stress Test Database
(PhysioNet ``nstdb``) into ``<repo>/data/physionet/nstdb/``.

What was actually found on PhysioNet (discovered programmatically, not
hardcoded from memory)
-----------------------------------------------------------------------
Fetching ``https://physionet.org/files/nstdb/1.0.0/RECORDS`` returns 15
records:

    118e00 118e06 118e12 118e18 118e24 118e_6
    119e00 119e06 119e12 119e18 119e24 119e_6
    bw  em  ma

The first twelve are *pre-noised ECG records* (MIT-BIH records 118/119
plus calibrated ``em`` noise at a known SNR — they are not pure noise).
The last three are the **pure half-hour noise records** used for this
experiment. Their categories come from the database's own documentation
file ``nstdb.txt``, whose first paragraph says (verbatim):

    "The directory includes noise records (`bw', `em', and `ma') that
     may be added to ECG records by `nst' ..."

and the PhysioNet project description states these were assembled from
intervals containing predominantly baseline wander (record ``bw``),
muscle/EMG artifact (record ``ma``) and electrode-motion artifact
(record ``em``).

This module therefore:
  1. fetches the RECORDS list from PhysioNet (``list_remote_records``),
  2. fetches ``nstdb.txt`` and parses the noise-record names out of the
     "noise records (...)" sentence (``parse_noise_record_names``),
  3. maps each discovered record name to its category with the
     documented name→category table below, failing loudly if PhysioNet
     ever serves a noise record name we do not recognise.

If the network is unavailable it falls back to the locally downloaded
``*.hea`` files, treating every record that is not a pre-noised ECG
record (``<rec>e<snr>`` naming scheme) as a pure noise record.
"""
import os
import re

import numpy as np

from load_real_ecg import (MITDB_DIR, DATA_ROOT, fetch_text,  # noqa: F401
                           list_remote_records, local_records,
                           records_dir, download_records, load_ecg_record,
                           REPO_ROOT)

NSTDB_DIR = records_dir("nstdb")
NSTDB_DOC_URL = "https://physionet.org/files/nstdb/1.0.0/nstdb.txt"

# Documented NSTDB record-name -> noise category (the names themselves
# are PhysioNet's abbreviations: bw = baseline wander, ma = muscle
# artifact, em = electrode motion artifact).
NOISE_CATEGORY_BY_RECORD = {
    "bw": "baseline_wander",
    "ma": "muscle_artifact",
    "em": "electrode_motion",
}
NOISE_CATEGORIES = ("baseline_wander", "muscle_artifact", "electrode_motion")

# Pre-noised ECG records look like 118e00 / 119e_6 (record + "e" + SNR).
_PRENOISED_ECG_RE = re.compile(r"^\d{3}e(_)?\d{2}$")
# "... noise records (`bw', `em', and `ma') that may be added ..."
_NOISE_DOC_RE = re.compile(r"noise records\s*\(([^)]*)\)", re.IGNORECASE)
_QUOTED_NAME_RE = re.compile(r"[`']([A-Za-z0-9_]+)[`']")


def parse_noise_record_names(doc_text):
    """Extract the pure-noise record names from the nstdb documentation
    text, i.e. from the sentence
    ``noise records (`bw', `em', and `ma') that may be added ...``.
    Returns a list in the order the names appear in the document."""
    match = _NOISE_DOC_RE.search(doc_text)
    if not match:
        raise ValueError("could not find the 'noise records (...)' "
                         "sentence in the nstdb documentation")
    return _QUOTED_NAME_RE.findall(match.group(1))


def _category_for(name):
    category = NOISE_CATEGORY_BY_RECORD.get(name)
    if category is None:
        raise ValueError(
            f"discovered NSTDB noise record {name!r} has no documented "
            f"category mapping (known: {NOISE_CATEGORY_BY_RECORD})")
    return category


def discover_noise_records():
    """Return ``{record_name: category}`` for the pure noise records of
    nstdb, discovered from PhysioNet's RECORDS + nstdb.txt."""
    try:
        all_records = list_remote_records("nstdb")
        doc = fetch_text(NSTDB_DOC_URL)
        doc_names = set(parse_noise_record_names(doc))
        noise_names = [r for r in all_records if r in doc_names]
        if not noise_names:
            raise ValueError("RECORDS and nstdb.txt did not agree on any "
                             "noise record")
    except Exception as exc:  # noqa: BLE001 - offline fallback
        print(f"  [info] remote NSTDB discovery failed ({exc}); "
              f"falling back to local files")
        local = local_records("nstdb")
        noise_names = [r for r in local if not _PRENOISED_ECG_RE.match(r)]
        if not noise_names:
            raise RuntimeError(
                f"nstdb is neither reachable nor downloaded — expected "
                f"records under {NSTDB_DIR}") from exc
    return {name: _category_for(name) for name in noise_names}


def noise_records_by_category():
    """Return ``{category: record_name}`` (e.g. {'baseline_wander': 'bw'})."""
    by_category = {}
    for name, category in discover_noise_records().items():
        by_category.setdefault(category, name)
    missing = [c for c in NOISE_CATEGORIES if c not in by_category]
    if missing:
        raise ValueError(f"no noise record found for categories {missing}")
    return by_category


def ensure_downloaded(records=None, dl_dir=None):
    """Download (a subset of) the nstdb noise records if they are not
    present yet. Annotation files are skipped — the pure noise records
    have none."""
    dl_dir = dl_dir or NSTDB_DIR
    wanted = records if records is not None else \
        list(discover_noise_records())
    missing = [r for r in wanted if not all(
        os.path.isfile(os.path.join(dl_dir, r + ext))
        for ext in (".hea", ".dat"))]
    if not missing:
        return dl_dir
    print(f"Downloading nstdb records {missing} -> {dl_dir} ...")
    download_records("nstdb", records=missing, dl_dir=dl_dir,
                     annotators=None)
    return dl_dir


def load_noise_record(name, channel=0, offset_s=0.0, duration_s=None):
    """Load one pure noise record (or raise a precise exception).

    Returns a dict: name, signal, fs, units, sig_name, category,
    offset_s, duration_s.
    """
    path = os.path.join(NSTDB_DIR, name)
    if not os.path.isfile(path + ".hea"):
        raise FileNotFoundError(
            f"nstdb noise record {name!r} not found in {NSTDB_DIR} — run "
            f"load_real_noise.ensure_downloaded() or run_real_experiment"
            f".py --download first")
    import wfdb  # local import keeps module import cheap for pure-text helpers
    try:
        record = wfdb.rdrecord(path)
    except Exception as exc:  # noqa: BLE001 - corrupted file
        raise RuntimeError(f"corrupted noise record {name!r} in "
                           f"{NSTDB_DIR}: {exc}") from exc

    fs = float(record.fs)  # from the WFDB header — never hardcoded
    if channel >= record.n_sig:
        raise ValueError(f"noise record {name!r} has {record.n_sig} "
                         f"channel(s); channel {channel} out of range")

    start = int(round(offset_s * fs))
    if start < 0:
        raise ValueError(f"offset_s must be >= 0, got {offset_s}")
    stop = record.sig_len if duration_s is None else \
        start + int(round(duration_s * fs))
    stop = min(stop, record.sig_len)
    if stop <= start:
        raise ValueError(f"noise record {name!r} has no samples in the "
                         f"requested window")

    signal = np.asarray(record.p_signal[start:stop, channel], dtype=float)
    bad = ~np.isfinite(signal)
    if bad.all():
        raise RuntimeError(f"noise record {name!r}: all samples invalid")
    if bad.any():
        idx = np.arange(signal.size)
        signal = signal.copy()
        signal[bad] = np.interp(idx[bad], idx[~bad], signal[~bad])
        print(f"  [warn] noise record {name!r}: replaced "
              f"{int(bad.sum())} invalid samples")

    try:
        category = NOISE_CATEGORY_BY_RECORD[name]
    except KeyError:
        raise ValueError(f"unknown noise record name {name!r}") from None

    return {
        "name": name,
        "signal": signal,
        "fs": fs,
        "units": str(record.units[channel]),
        "sig_name": str(record.sig_name[channel]),
        "category": category,
        "offset_s": offset_s,
        "duration_s": (stop - start) / fs,
    }


def load_noise_records(names, **kwargs):
    """Load several noise records, skipping bad ones with a message."""
    loaded = []
    for name in names:
        try:
            loaded.append(load_noise_record(name, **kwargs))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"  [skip] noise record {name!r}: {exc}")
    return loaded


# Convenience re-exports so the experiment runner can import everything
# it needs for data handling from one place.
__all__ = [
    "NSTDB_DIR", "NOISE_CATEGORY_BY_RECORD", "NOISE_CATEGORIES",
    "parse_noise_record_names", "discover_noise_records",
    "noise_records_by_category", "ensure_downloaded",
    "load_noise_record", "load_noise_records",
    "MITDB_DIR", "DATA_ROOT", "REPO_ROOT", "list_remote_records",
    "local_records", "download_records", "load_ecg_record",
]
