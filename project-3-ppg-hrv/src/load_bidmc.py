"""
Loader for the BIDMC PPG and Respiration Database (PhysioNet).

Dataset reference
-----------------
Silas P. Johnson, M.D. et al. "The Multiparameter Intelligent Database in
Intensive Care (MIMIC-II): A freely accessible critical care database."
and the BIDMC PPG and Respiration Database described at
https://physionet.org/content/bidmc/1.0.0/

Contents observed from the actual WFDB headers (see _EXPECTED_* below):

* Waveform records ``bidmc01`` .. ``bidmc53`` (53 recordings, ~8 min
  each): sampling frequency read from the header (125 Hz in this
  database, but NEVER hardcoded here), signals are an impedance
  respiration channel, a PPG (photoplethysmogram) channel and one or
  more ECG leads.  The channel ORDER AND COUNT VARY BETWEEN RECORDS
  (observed: 5 or 6 signals; ``PLETH`` appears at index 1, 2 or 4; some
  records carry an extra ``CVP`` channel), so channels are identified
  from header metadata (signal name + units), never by fixed index.
* Numerics records ``bidmc01n`` .. ``bidmc53n``: 1 Hz monitor-derived
  numerics (HR, PULSE, RESP, SpO2).  These are MONITOR-DERIVED
  REFERENCES produced by the bedside monitor's own (undocumented here)
  algorithm -- they are NOT manual beat annotations.  BIDMC provides no
  manual beat annotations at all.

All downloaded files live under the repository root in
``data/physionet/bidmc/`` (git-ignored; never committed).
"""
import json
import os

import numpy as np
import wfdb

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
REPO_ROOT = os.path.dirname(_PROJECT_DIR)
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "physionet", "bidmc")

N_RECORDS = 53  # bidmc01 .. bidmc53

#: Records the real experiment runs on by default (README: "3 records,
#: ``bidmc01``–``bidmc03``").  The database contains :data:`N_RECORDS`
#: (53) recordings; restricting the default keeps the committed
#: ``outputs/real/`` artefacts reproducible from a plain
#: ``python run_bidmc_experiment.py --download``.  Pass ``records=``
#: explicitly to analyse others.
DEFAULT_RECORDS = ["bidmc01", "bidmc02", "bidmc03"]

# Channel-name conventions observed in the real headers (documented here
# for reference only -- identification is done by matching, see
# identify_channels()).
WAVEFORM_NAMES_OBSERVED = ["RESP", "PLETH", "II", "V", "AVR", "CVP"]
NUMERICS_NAMES_OBSERVED = ["HR", "PULSE", "RESP", "SpO2"]

# Standard ECG lead names, in preference order for R-peak detection
# (lead II is the classic choice for QRS detection).
_ECG_LEAD_PREFERENCE = ["ii", "iii", "i", "v", "avr", "avl", "avf", "ecg"]

# Numerics role identification (case-insensitive exact / substring match)
_NUMERIC_ROLES = {
    "hr": ["hr", "heart rate", "heartrate"],
    "pulse": ["pulse", "pulse rate"],
    "resp": ["resp", "respiratory rate", "rr"],
    "spo2": ["spo2", "sao2", "oxygen saturation"],
}


def _norm_name(s):
    """Normalize a WFDB signal name for matching.

    Real headers in this database store names with a trailing comma
    (observed: ``'RESP,'``, ``'II,'``), so strip whitespace and commas
    and lowercase before matching.
    """
    return str(s).strip().strip(",").strip().lower()


def identify_channels(sig_names, units=None):
    """Identify PPG / ECG / respiration channels from WFDB header metadata.

    Never assumes a fixed channel order: matches signal names
    case-insensitively against the naming conventions actually observed
    in this database (``PLETH``/``PPG`` for the pulse oximetry waveform,
    standard ECG lead names for the ECG, ``RESP`` for impedance
    respiration), using units as a secondary hint.

    Parameters
    ----------
    sig_names : sequence of str
        ``record.sig_name`` from the WFDB header.
    units : sequence of str, optional
        ``record.units`` from the WFDB header.

    Returns
    -------
    dict with keys ``ppg``, ``ecg``, ``resp`` (channel index or None)
    and ``ecg_lead`` (chosen ECG lead name or None).
    """
    sig_names = [str(s) for s in sig_names]
    units = [str(u) for u in units] if units is not None else [None] * len(sig_names)
    lower = [_norm_name(s) for s in sig_names]

    ppg_idx = None
    resp_idx = None
    ecg_candidates = []

    for i, name in enumerate(lower):
        unit = (units[i] or "").strip().lower()
        # --- PPG: pulse oximetry photoplethysmogram -------------------
        if ppg_idx is None and (
            name in ("pleth", "ppg", "plethysmogram", "plethysmograph")
            or "pleth" in name
            or name.startswith("ppg")
        ):
            ppg_idx = i
            continue
        # --- respiration (impedance) -----------------------------------
        if resp_idx is None and "resp" in name and "rate" not in name:
            resp_idx = i
            continue
        # --- ECG: standard lead names, optionally confirmed by mV units
        if name in _ECG_LEAD_PREFERENCE or "ecg" in name:
            if unit in ("mv", ""):  # ECG is in mV; skip if units disagree
                ecg_candidates.append(i)

    ecg_idx = None
    ecg_lead = None
    for lead in _ECG_LEAD_PREFERENCE:  # preference order, not header order
        for i in ecg_candidates:
            if lower[i] == lead:
                ecg_idx, ecg_lead = i, sig_names[i]
                break
        if ecg_idx is not None:
            break
    if ecg_idx is None and ecg_candidates:
        ecg_idx, ecg_lead = ecg_candidates[0], sig_names[ecg_candidates[0]]
    if ecg_lead is not None:
        ecg_lead = ecg_lead.strip().strip(",").strip()

    return {"ppg": ppg_idx, "ecg": ecg_idx, "resp": resp_idx,
            "ecg_lead": ecg_lead}


def identify_numerics(sig_names, units=None):
    """Map numerics-record channels to roles {hr, pulse, resp, spo2}.

    Returns a dict role -> channel index (or None when absent).
    """
    sig_names = [str(s) for s in sig_names]
    lower = [_norm_name(s) for s in sig_names]
    out = {}
    for role, keys in _NUMERIC_ROLES.items():
        idx = None
        for i, name in enumerate(lower):
            if name in keys or any(name == k for k in keys):
                idx = i
                break
        if idx is None:  # substring fallback
            for i, name in enumerate(lower):
                if any(k in name for k in keys):
                    idx = i
                    break
        out[role] = idx
    return out


def _count_invalid(arr):
    """Count NaN / inf values in an array (returns nan count, inf count)."""
    a = np.asarray(arr, dtype=float)
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    return n_nan, n_inf


def list_local_records(data_dir=DEFAULT_DATA_DIR):
    """Return sorted record names (e.g. ['bidmc01', ...]) present locally."""
    if not os.path.isdir(data_dir):
        return []
    names = []
    for f in os.listdir(data_dir):
        if f.endswith(".hea") and not f.endswith("n.hea"):
            names.append(f[:-4])
    return sorted(names)


def download_records(records=None, data_dir=DEFAULT_DATA_DIR, overwrite=False):
    """Programmatically download BIDMC records with wfdb.

    Parameters
    ----------
    records : list of str or None
        Record names, e.g. ``['bidmc01', 'bidmc02']``; None downloads
        :data:`DEFAULT_RECORDS` (``bidmc01``–``bidmc03``, the subset the
        real experiment reports on).  The full database is
        ``["bidmc%02d" % i for i in range(1, N_RECORDS + 1)]``.
    data_dir : str
        Destination directory (repo-root ``data/physionet/bidmc``).

    Returns
    -------
    list of str : record names present locally after the download.
    """
    os.makedirs(data_dir, exist_ok=True)
    if records is None:
        records = list(DEFAULT_RECORDS)
    # The 1 Hz numerics are separate WFDB records (bidmc01n, ...) that are
    # NOT listed in the database RECORDS file, so request them explicitly.
    expanded = []
    for r in records:
        expanded.append(r)
        if not r.endswith("n"):
            expanded.append(r + "n")
    wfdb.dl_database("bidmc", data_dir, records=expanded,
                     annotators=[], overwrite=overwrite)
    return list_local_records(data_dir)


def load_record(record_name, data_dir=DEFAULT_DATA_DIR,
                include_numerics=True, strict=False):
    """Load one BIDMC record: waveforms + (optional) 1 Hz monitor numerics.

    Channel discovery, sampling frequency and lengths all come from the
    WFDB headers -- nothing is hardcoded.  Missing channels are reported
    in ``warnings`` and the corresponding entries are None (unless
    ``strict=True``, which raises instead).

    Returns
    -------
    dict
        record_name, fs, sig_len, duration_s, sig_names, units,
        ppg / ecg / resp (np.ndarray or None), ppg_channel / ecg_channel /
        resp_channel / ecg_lead, channels (identify_channels result),
        numerics (dict of monitor-derived reference series or None),
        n_nan / n_inf per channel, warnings, hea_path.
    """
    hea_path = os.path.join(data_dir, record_name + ".hea")
    if not os.path.isfile(hea_path):
        raise FileNotFoundError(
            "BIDMC record '%s' not found at %s. Download it first, e.g. "
            "python -c \"import wfdb; wfdb.dl_database('bidmc', %r, "
            "records=['%s'])\"" % (record_name, hea_path, data_dir, record_name))

    rec = wfdb.rdrecord(hea_path[:-4])  # rdrecord appends .hea itself
    fs = float(rec.fs)                  # sampling frequency from the header
    sig_len = int(rec.sig_len)
    warnings = []
    # clean header names (real headers carry trailing commas)
    rec_sig_name = [str(s).strip().strip(",").strip() for s in rec.sig_name]

    if fs <= 0:
        raise ValueError("Record %s has non-positive fs=%r in header"
                         % (record_name, fs))

    ch = identify_channels(rec_sig_name, rec.units)
    if ch["ppg"] is None:
        warnings.append("no PPG channel found (signals: %s)"
                        % ", ".join(rec_sig_name))
        if strict:
            raise ValueError(warnings[-1])
    if ch["ecg"] is None:
        warnings.append("no ECG channel found (signals: %s)"
                        % ", ".join(rec_sig_name))
        if strict:
            raise ValueError(warnings[-1])

    def _chan(idx):
        if idx is None:
            return None
        x = np.asarray(rec.p_signal[:, idx], dtype=float)
        if len(x) != sig_len:
            warnings.append("channel %s length %d != header sig_len %d"
                            % (rec_sig_name[idx], len(x), sig_len))
        return x

    ppg = _chan(ch["ppg"])
    ecg = _chan(ch["ecg"])
    resp = _chan(ch["resp"])

    n_nan = {}
    n_inf = {}
    for role, arr in (("ppg", ppg), ("ecg", ecg), ("resp", resp)):
        if arr is None:
            n_nan[role] = n_inf[role] = None
        else:
            n_nan[role], n_inf[role] = _count_invalid(arr)
            if n_nan[role] or n_inf[role]:
                warnings.append("%s channel has %d NaN / %d inf samples"
                                % (role, n_nan[role], n_inf[role]))

    # ---------------- 1 Hz monitor-derived numerics --------------------
    numerics = None
    num_hea = os.path.join(data_dir, record_name + "n.hea")
    if include_numerics and os.path.isfile(num_hea):
        nrec = wfdb.rdrecord(num_hea[:-4])
        n_num_name = [str(s).strip().strip(",").strip()
                      for s in nrec.sig_name]
        n_roles = identify_numerics(n_num_name, nrec.units)
        t_num = np.arange(int(nrec.sig_len)) / float(nrec.fs)
        numerics = {
            "fs": float(nrec.fs),
            "t": t_num,
            "sig_len": int(nrec.sig_len),
            "sig_name": n_num_name,
            "roles": n_roles,
            "warning": ("MONITOR-DERIVED REFERENCES (bedside monitor "
                        "algorithm), not manual beat annotations"),
        }
        for role, idx in n_roles.items():
            if idx is None:
                numerics[role] = None
                warnings.append("numerics record has no %s channel" % role)
                continue
            vals = np.asarray(nrec.p_signal[:, idx], dtype=float)
            nn, ni = _count_invalid(vals)
            # monitor often uses 0 / negative as "no reading" sentinels
            if role in ("hr", "pulse", "resp"):
                vals = np.where(vals <= 0, np.nan, vals)
            if nn or ni:
                warnings.append("numerics %s: %d NaN / %d inf (before "
                                "sentinel masking)" % (role, nn, ni))
            numerics[role] = vals
        # length consistency: waveform duration vs numerics duration
        dur_wf = sig_len / fs
        dur_num = numerics["sig_len"] / numerics["fs"]
        if abs(dur_wf - dur_num) > 2.0:
            warnings.append("waveform duration %.1f s != numerics duration "
                            "%.1f s" % (dur_wf, dur_num))
    elif include_numerics:
        warnings.append("numerics record %sn.hea not found locally"
                        % record_name)

    return {
        "record_name": record_name,
        "fs": fs,
        "sig_len": sig_len,
        "duration_s": sig_len / fs,
        "sig_names": rec_sig_name,
        "units": list(rec.units),
        "channels": ch,
        "ppg_channel": rec_sig_name[ch["ppg"]] if ch["ppg"] is not None else None,
        "ecg_channel": rec_sig_name[ch["ecg"]] if ch["ecg"] is not None else None,
        "resp_channel": rec_sig_name[ch["resp"]] if ch["resp"] is not None else None,
        "ecg_lead": ch["ecg_lead"],
        "ppg": ppg, "ecg": ecg, "resp": resp,
        "n_nan": n_nan, "n_inf": n_inf,
        "numerics": numerics,
        "warnings": warnings,
        "hea_path": hea_path,
    }


def save_record_metadata(bundle, out_dir):
    """Write per-record metadata JSON (no waveform data) to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    meta = {k: v for k, v in bundle.items()
            if k not in ("ppg", "ecg", "resp", "numerics")}
    numerics = bundle.get("numerics")
    if numerics is not None:
        # keep metadata only (no bulky reference arrays)
        meta["numerics"] = {k: v for k, v in numerics.items()
                            if k not in ("t", "roles") and not isinstance(v, np.ndarray)}
        meta["numerics"]["roles"] = {
            r: (None if i is None else numerics["sig_name"][i])
            for r, i in numerics["roles"].items()}
    # make JSON-safe
    def _safe(o):
        if isinstance(o, dict):
            return {str(k): _safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    path = os.path.join(out_dir, bundle["record_name"] + ".json")
    with open(path, "w") as f:
        json.dump(_safe(meta), f, indent=2)
    return path
