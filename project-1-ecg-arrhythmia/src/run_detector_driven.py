"""Detector-driven classification evaluation (P1.3).

The *annotation-centred* benchmark (``run_mitbih_experiment --mode mitbih``)
trains a classifier on features built from expert ``.atr`` beat
annotations — retrospective by design.  This script adds a **detector-driven
evaluation**:

1. Run the from-scratch R-peak detector (``detect_rpeaks``) on the ECG.
2. Match detected peaks to expert annotations with **one-to-one matching**
   (``match_peaks_one_to_one``, ±50 ms).
3. **Assign labels only from the matched reference annotations**
   (``map_to_aami_class``).  Reference annotation *positions* are never used
   to create detector-driven input features.
4. Extract **causal morphology features** around detected peaks
   (``extract_qrs_morphology_features``) and **causal RR features using
   only past intervals** (``_causal_rr_features``).
5. Train RF / LR and evaluate with per-class metrics, confusion matrix,
   and a confusion figure.

Report (``outputs/detector_driven/metrics.json``):
- Matched beats (TP), unmatched detections (FP), missed reference beats
  (FN).
- Classification metrics (balanced accuracy, macro-F1, per-class) on
  matched detections.
- Combined end-to-end detection + classification limitations.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "detector_driven")

from aami import get_aami_classes, map_to_aami_class
from detect_rpeaks import detect_rpeaks, match_peaks_one_to_one
from features import extract_qrs_morphology_features
from load_mitbih import MITBIHLoader

AAMI_CLASSES = get_aami_classes()  # ["N", "S", "V", "F"]

FS_TARGET = 360
TOL_MS = 50.0  # predefined one-to-one matching tolerance


def _causal_rr_features(det_peaks: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    """QRS/RR-context features using only *past* intervals (causal: no
    future-dependent features).  Row ``k`` corresponds to detected peak
    ``det_peaks[k+1]`` (interior peaks)."""
    peaks = np.asarray(det_peaks, dtype=int)
    if len(peaks) < 3:
        return np.empty((0, 3)), ["rr_prev", "rr_mean_past", "rr_prev_over_past"]
    rr = np.diff(peaks) / float(fs)          # rr[k] = peak[k+1] - peak[k]
    rows = []
    for j in range(1, len(peaks) - 1):
        rr_prev = rr[j - 1]                   # interval ending at peak j
        past = rr[max(0, j - 6):j - 1]        # strictly past (excludes the
                                               # interval ending at peak j)
        rr_mean_past = float(np.mean(past)) if len(past) else rr_prev
        rows.append([rr_prev, rr_mean_past, rr_prev / max(rr_mean_past, 1e-9)])
    return np.asarray(rows, dtype=float), \
        ["rr_prev", "rr_mean_past", "rr_prev_over_past"]


def _matched_features(signal, det_peaks, label_rows, fs):
    """Features for matched detected peaks only.

    ``label_rows`` maps *detected-peak index* → AAMI class (labels come
    only from matched reference annotations — never from detector inputs).

    Returns ``(X, y, feature_names, n_unmatched, n_missed_ref)``.
    """
    det_peaks = np.asarray(det_peaks, dtype=int)
    morph, morph_names = extract_qrs_morphology_features(signal, det_peaks, fs)
    rr_f, rr_names = _causal_rr_features(det_peaks, fs)
    names = list(morph_names) + list(rr_names)

    n_det = det_peaks.size
    # Detections (by index) with no label:
    n_unmatched = int(sum(1 for j in range(1, n_det) if j not in label_rows))

    X, y = [], []
    for j in range(1, n_det - 1):
        k = j - 1
        if j in label_rows:
            morph_row = morph[k] if k < morph.shape[0] else np.zeros(len(morph_names))
            rr_row = rr_f[k] if k < rr_f.shape[0] else np.zeros(len(rr_names))
            X.append(np.hstack([morph_row, rr_row]))
            y.append(label_rows[j])
    X = np.asarray(X, dtype=float).reshape(-1, len(names)) if X else \
        np.empty((0, len(names)))
    y = np.asarray(y, dtype=object)
    n_matched = y.size
    return X, y, names, n_unmatched, n_matched


def _classification_report(y_true, y_pred, class_names):
    out = {}
    if y_true.size == 0:
        return {"balanced_accuracy": float("nan"), "macro_f1": float("nan"),
                "per_class": {}}
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, labels=class_names,
                                     average="macro", zero_division=0))
    out["per_class"] = {}
    for c in class_names:
        mask = y_true == c
        tp = int(np.sum((y_pred == c) & mask))
        fp = int(np.sum((y_pred == c) & ~mask))
        fn = int(np.sum((y_pred != c) & mask))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out["per_class"][c] = {"precision": round(prec, 4),
                               "recall": round(rec, 4),
                               "f1": round(f1, 4), "support": int(mask.sum())}
    return out


def _plot_confusion(y_true, y_pred, class_names, path, record):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{record}: detector-driven confusion (matched beats)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    return cm


def run_detector_driven_eval(loader, train_records, val_records, test_records,
                             fs_target=FS_TARGET, tol_ms=TOL_MS,
                             random_state=42):
    """Detect → match (one-to-one, ±tol_ms) → label from matched
    annotations only → classify with causal features."""
    def _prepare(records):
        Xs, ys, stats = [], [], []
        for rec in records:
            b = loader.load_record(rec)
            signal = np.asarray(b["signal"], dtype=float)
            ref_peaks = np.asarray(b["r_peaks"], dtype=int)
            ann_symbols = np.asarray(b["labels"], dtype=str)
            det = detect_rpeaks(signal, b["fs"])
            pairs = match_peaks_one_to_one(det, ref_peaks, b["fs"],
                                           tol_ms / 1000.0)
            label_rows = {}
            for det_idx, ref_idx, _err_ms in pairs:
                if ref_idx < len(ann_symbols):
                    aami_cls = map_to_aami_class(ann_symbols[ref_idx])
                    if aami_cls in AAMI_CLASSES:
                        label_rows[int(det_idx)] = aami_cls
            matched_refs = {ref_idx for _d, ref_idx, _e in pairs
                            if ref_idx < len(ann_symbols)}
            X, y, names, n_unmatched, _n_matched = _matched_features(
                signal, det, label_rows, b["fs"])
            Xs.append(X); ys.append(y)
            stats.append({"record": rec,
                          "n_detected": int(det.size),
                          "n_matched": int(y.size),
                          "n_unmatched_detections_fp": n_unmatched,
                          "n_reference": int(ref_peaks.size),
                          "n_missed_reference_fn": int(ref_peaks.size - len(matched_refs)),
                          "feature_names": names})
        X = np.vstack(Xs) if Xs and Xs[0].size else np.empty((0, 0))
        y = np.concatenate(ys) if ys and ys[0].size else np.array([], dtype=object)
        return X, y, stats

    X_train, y_train, stats_train = _prepare(train_records)
    X_test, y_test, stats_test = _prepare(test_records)
    report = {"tolerance_ms": float(tol_ms), "fs_target": float(fs_target),
              "train_records": list(train_records), "test_records": list(test_records),
              "per_record": stats_train + stats_test, "models": {}}

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        report["error"] = "no matched beats in train/test records"
        return report
    report["feature_names"] = (stats_train[0]["feature_names"] if stats_train else [])
    report["n_features"] = int(X_train.shape[1])
    report["test_summary"] = {
        "n_test_matched": int(y_test.size),
        "test_class_counts": {c: int(np.sum(y_test == c)) for c in AAMI_CLASSES},
        "n_test_unmatched_detections_fp": int(sum(s["n_unmatched_detections_fp"]
                                                  for s in stats_test)),
        "n_test_missed_reference_fn": int(sum(s["n_missed_reference_fn"]
                                              for s in stats_test)),
    }

    for name, cls in (("random_forest", RandomForestClassifier(
                          n_estimators=200, max_depth=15, random_state=random_state,
                          class_weight="balanced")),
                      ("logistic_regression", LogisticRegression(
                          C=1.0, max_iter=1000, random_state=random_state))):
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        report["models"][name] = _classification_report(y_test, y_pred, AAMI_CLASSES)
        if name == "random_forest":
            cm = _plot_confusion(y_test, y_pred, AAMI_CLASSES,
                                 os.path.join(OUT_DIR, "confusion_detector_driven.png"),
                                 ",".join(test_records))
            report["models"][name]["confusion_matrix"] = cm.tolist()
    return report


def main():
    from evaluation import get_classic_split
    os.makedirs(OUT_DIR, exist_ok=True)
    loader = MITBIHLoader()
    classic = get_classic_split()
    report = run_detector_driven_eval(
        loader, classic.train_records, classic.val_records, classic.test_records)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    for k, v in report["models"].items():
        print(f"{k}: bal_acc={v['balanced_accuracy']:.4f} macro_f1={v['macro_f1']:.4f}")
    print("saved ->", os.path.join(OUT_DIR, "metrics.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
