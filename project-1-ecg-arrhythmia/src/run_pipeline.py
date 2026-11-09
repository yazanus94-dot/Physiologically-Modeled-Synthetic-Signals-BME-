"""
End-to-end ECG arrhythmia beat-classification pipeline:

generate -> R-peak detect -> extract features -> Random Forest
-> classification report + confusion matrix + feature importance
-> plots + metrics.json in outputs/.

Run:  cd src && python run_pipeline.py
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from generate_signal import generate_ecg
from detect_rpeaks import detect_rpeaks, score_detector
from features import extract_features

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("1. Generating 10 min synthetic ECG at 250 Hz (N/V/S beats + noise)...")
    signal, r_true, labels, fs = generate_ecg(duration_s=600, fs=250, seed=42)
    print(f"   {len(r_true)} beats: "
          f"N={labels.count('N')}  V={labels.count('V')}  S={labels.count('S')}")

    print("2. Detecting R peaks (Pan-Tompkins style)...")
    r_det = detect_rpeaks(signal, fs)
    det_score = score_detector(r_det, r_true, fs, tolerance_s=0.05)
    print(f"   sensitivity={det_score['sensitivity']:.4f}  "
          f"precision={det_score['precision']:.4f}  "
          f"(TP={det_score['tp']} FP={det_score['fp']} FN={det_score['fn']})")

    print("3. Extracting beat-level features from detected peaks...")
    r_det_arr = np.asarray(r_det, dtype=int)
    r_true_arr = np.asarray(r_true, dtype=int)
    # align each detected beat to its ground-truth label (unmatched
    # detections are dropped from the classification stage)
    tol = int(0.05 * fs)
    y_det, keep_rows = [], []
    for row, p in enumerate(r_det_arr):
        if row == 0 or row == len(r_det_arr) - 1:
            continue  # features.py skips the first/last beat anyway
        j = int(np.argmin(np.abs(r_true_arr - p)))
        if abs(int(p) - int(r_true_arr[j])) <= tol:
            y_det.append(labels[j])
            keep_rows.append(row)
    X, feat_names = extract_features(signal, r_det, fs)
    # feature row f corresponds to detected-peak row f+1 (features.py skips
    # the first and last detected peak)
    X = X[[k - 1 for k in keep_rows]]
    y = np.asarray(y_det)
    print(f"   X={X.shape}  y={y.shape}  classes={sorted(set(y))}")

    print("4. Training / evaluating the Random Forest classifier...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    report_txt = classification_report(y_te, clf.predict(X_te))
    cm = confusion_matrix(y_te, clf.predict(X_te), labels=["N", "V", "S"])
    class_names = ["N", "V", "S"]
    report_dict = classification_report(y_te, clf.predict(X_te), output_dict=True)

    print("\n" + report_txt)
    print("Confusion matrix (rows=true N/V/S, cols=predicted N/V/S):")
    print(cm)

    # --- persist model + metrics ---
    joblib.dump(clf, os.path.join(OUT_DIR, "ecg_beat_classifier.joblib"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        import json
        json.dump({
            "n_beats": len(r_true),
            "beat_counts": {c: labels.count(c) for c in ["N", "V", "S"]},
            "r_peak_detection": det_score,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "feature_names": feat_names,
            "feature_importances": dict(zip(feat_names, [round(float(v), 4) for v in clf.feature_importances_])),
        }, f, indent=2)
    print("Saved model + metrics -> outputs/ecg_beat_classifier.joblib, outputs/metrics.json")

    # --- Plot 1: signal strip with detected peaks ---
    dur = 8
    n = dur * fs
    t = np.arange(n) / fs
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(t, signal[:n], color="#1B3A6B", lw=0.8)
    det_in = r_det[r_det < n]
    true_in = r_true[r_true < n]
    ax.plot(t[det_in], signal[det_in], "g^", ms=8, label="detected R peaks")
    ax.plot(t[true_in], signal[true_in] - 0.06, "xr", ms=8, mew=1.5, label="ground truth")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"First {dur} s of synthetic ECG — R-peak detection\n"
                 f"sensitivity={det_score['sensitivity']:.3f}, precision={det_score['precision']:.3f}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ecg_with_detected_peaks.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: confusion matrix ---
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(class_names)
    ax.set_yticks(range(3)); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Beat classification confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: feature importance ---
    fig, ax = plt.subplots(figsize=(6, 3.4))
    order = np.argsort(clf.feature_importances_)[::-1]
    ax.bar([feat_names[i] for i in order],
           [clf.feature_importances_[i] for i in order], color="#2563AE")
    ax.set_ylabel("importance")
    ax.set_title("Random Forest feature importances")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)

    print("Saved plots -> outputs/ecg_with_detected_peaks.png, "
          "confusion_matrix.png, feature_importance.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
