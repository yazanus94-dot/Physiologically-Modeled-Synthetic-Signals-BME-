"""
End-to-end ECG arrhythmia beat-classification pipeline.

Supports two experimental modes:
1. Synthetic experiment (existing) - uses physiologically-informed synthetic ECG
2. Real MIT-BIH experiment - uses actual PhysioNet MIT-BIH recordings with AAMI 4-class classification

Run synthetic:    cd src && python run_pipeline.py --mode synthetic
Run real data:    cd src && python run_pipeline.py --mode mitbih --protocol classic
Run real data:    cd src && python run_pipeline.py --mode mitbih --protocol leakage_audited
"""
import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from generate_signal import generate_ecg
from detect_rpeaks import (
    detect_rpeaks, detect_rpeaks_mitbih, score_detector, score_detector_detailed,
    match_peaks_one_to_one,
)
from features import extract_features, extract_features_from_beats
from load_mitbih import MITBIHLoader
from aami import (
    map_to_aami_class, filter_aami_beats,
    segment_beats_annotation_centered, count_beats_per_class
)
from preprocessing import preprocess_ecg, PreprocessingConfig
from evaluation import (
    get_classic_split, get_leakage_audited_split,
    collect_beats_from_records, verify_no_leakage
)
from train_model import (
    ModelConfig, train_model, evaluate_model,
    evaluate_per_record, get_feature_importance, save_model
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "outputs"
SYNTHETIC_OUT = OUT_DIR / "synthetic"
MITBIH_OUT = OUT_DIR / "mitbih"

AAMI_CLASSES = ["N", "S", "V", "F"]


def run_synthetic_experiment(seed: int = 42, duration_s: float = 600.0):
    """Run the synthetic ECG experiment (original pipeline)."""
    print("=" * 60)
    print("SYNTHETIC ECG EXPERIMENT")
    print("=" * 60)

    SYNTHETIC_OUT.mkdir(parents=True, exist_ok=True)

    print(f"1. Generating {duration_s/60:.0f} min synthetic ECG at 250 Hz (N/V/S beats + noise)...")
    signal, r_true, labels, fs = generate_ecg(duration_s=duration_s, fs=250, seed=seed)
    print(f"   {len(r_true)} beats: "
          f"N={labels.count('N')}  V={labels.count('V')}  S={labels.count('S')}")

    print("2. Detecting R peaks (Pan-Tompkins style)...")
    r_det = detect_rpeaks(signal, fs, lowcut=2.0, highcut=15.0, use_sos=False)
    det_score = score_detector_detailed(r_det, r_true, fs, tolerance_s=0.05)
    print(f"   sensitivity={det_score['sensitivity']:.4f}  "
          f"precision={det_score['precision']:.4f}  "
          f"f1={det_score['f1']:.4f}")
    print(f"   TP={det_score['tp']} FP={det_score['fp']} FN={det_score['fn']}")
    print(f"   Mean timing error: {det_score['mean_timing_error_ms']:.2f} ms")

    print("3. Extracting beat-level features from detected peaks...")
    r_det_arr = np.asarray(r_det, dtype=int)
    r_true_arr = np.asarray(r_true, dtype=int)
    # One-to-one match detections to ground truth (so no two detections can
    # inherit the same reference beat's class label), then keep only the
    # detections that extract_features() actually produced a row for —
    # it skips the first and last peak.
    pairs = match_peaks_one_to_one(r_det_arr, r_true_arr, fs, tolerance_s=0.05)
    last_row = len(r_det_arr) - 1
    kept = [(j, i) for (j, i, _e) in pairs if 0 < j < last_row]
    keep_rows = [j for (j, _i) in kept]
    y_det = [labels[i] for (_j, i) in kept]
    X, feat_names = extract_features(signal, r_det, fs)
    X = X[[k - 1 for k in keep_rows]]
    y = np.asarray(y_det)
    print(f"   X={X.shape}  y={y.shape}  classes={sorted(set(y))}")

    print("4. Training / evaluating the Random Forest classifier...")
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)

    config = ModelConfig(model_type="random_forest", n_estimators=200)
    model, train_info = train_model(X_tr, y_tr, config)

    eval_result = evaluate_model(model, X_te, y_te, class_names=["N", "V", "S"])

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Synthetic)")
    print("=" * 60)
    from sklearn.metrics import classification_report
    # `labels` must be passed explicitly: target_names is paired positionally
    # with it, and sklearn would otherwise sort the observed classes and map
    # ['N','V','S'] onto ['N','S','V'], printing the wrong class names.
    class_order = ["N", "V", "S"]
    print(classification_report(y_te, model.predict(X_te),
                                labels=class_order,
                                target_names=class_order,
                                zero_division=0))
    print(f"Confusion matrix (rows=true {class_order}, "
          f"cols=predicted {class_order}):")
    print(eval_result["confusion_matrix"])

    # Save model + metrics
    joblib.dump(model, SYNTHETIC_OUT / "ecg_beat_classifier.joblib")
    with open(SYNTHETIC_OUT / "metrics.json", "w") as f:
        json.dump({
            "experiment": "synthetic",
            "seed": seed,
            "duration_s": duration_s,
            "n_beats": len(r_true),
            "beat_counts": {c: labels.count(c) for c in ["N", "V", "S"]},
            "r_peak_detection": det_score,
            "classification_report": eval_result["classification_report"],
            "confusion_matrix": eval_result["confusion_matrix"],
            "balanced_accuracy": eval_result["balanced_accuracy"],
            "macro_f1": eval_result["macro_f1"],
            "weighted_f1": eval_result["weighted_f1"],
            "feature_names": feat_names,
            "feature_importances": get_feature_importance(model, feat_names),
        }, f, indent=2)
    print(f"Saved model + metrics -> {SYNTHETIC_OUT}")

    # Plots
    _plot_synthetic_results(signal, r_true, r_det, det_score, eval_result, feat_names, model, fs, SYNTHETIC_OUT)

    return eval_result


def _plot_synthetic_results(signal, r_true, r_det, det_score, eval_result, feat_names, model, fs, out_dir):
    """Generate plots for synthetic experiment."""
    # Plot 1: signal strip with detected peaks
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
    fig.savefig(out_dir / "ecg_with_detected_peaks.png", dpi=150)
    plt.close(fig)

    # Plot 2: confusion matrix
    cm = np.array(eval_result["confusion_matrix"])
    class_names = ["N", "V", "S"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(class_names)
    ax.set_yticks(range(3)); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Beat classification confusion matrix (Synthetic)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Plot 3: feature importance
    importances = get_feature_importance(model, feat_names)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    order = np.argsort([importances.get(f, 0) for f in feat_names])[::-1]
    ax.bar([feat_names[i] for i in order],
           [importances.get(feat_names[i], 0) for i in order], color="#2563AE")
    ax.set_ylabel("importance")
    ax.set_title("Random Forest feature importances (Synthetic)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots -> {out_dir}")


def run_mitbih_experiment(protocol: str = "classic", fs_target: int = 360):
    """
    Run the real MIT-BIH experiment with AAMI 4-class classification.

    Parameters
    ----------
    protocol : str
        "classic" or "leakage_audited"
    fs_target : int
        Target sampling frequency for resampling (default 360 Hz, native MIT-BIH rate)
    """
    print("=" * 60)
    print(f"MIT-BIH EXPERIMENT (protocol: {protocol})")
    print("=" * 60)

    MITBIH_OUT.mkdir(parents=True, exist_ok=True)

    # Get evaluation split
    if protocol == "classic":
        split = get_classic_split()
    elif protocol == "leakage_audited":
        split = get_leakage_audited_split()
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    print(f"Split: {split}")
    verify_no_leakage(split.train_records, split.test_records)

    # Initialize loader
    loader = MITBIHLoader()

    # Check available records
    available = loader.get_available_records()
    print(f"Available records: {len(available)}/48")

    # Filter splits to available records
    train_records = [r for r in split.train_records if r in available]
    val_records = [r for r in split.val_records if r in available]
    test_records = [r for r in split.test_records if r in available]

    print(f"Train records ({len(train_records)}): {train_records}")
    print(f"Val records ({len(val_records)}): {val_records}")
    print(f"Test records ({len(test_records)}): {test_records}")

    if not train_records or not test_records:
        raise RuntimeError("Insufficient records available. Download MIT-BIH data first.")

    # Preprocessing config
    preprocess_config = PreprocessingConfig(
        lowcut_hz=0.5,
        highcut_hz=40.0,
        filter_order=4,
        remove_dc=True,
        dc_method="median",
        normalize=False
    )

    print("\n1. Collecting training beats...")
    X_train, y_train, train_rec_ids, _, feat_names = collect_beats_from_records(
        loader, train_records, fs_target=fs_target, preprocess_config=preprocess_config
    )
    print(f"   Training: X={X_train.shape}, classes={np.unique(y_train)}")
    print(f"   Features ({len(feat_names)}): {feat_names}")
    train_counts = count_beats_per_class(y_train.tolist())
    print(f"   Class counts: {train_counts}")

    print("\n2. Collecting validation beats...")
    X_val, y_val, val_rec_ids, _, _ = collect_beats_from_records(
        loader, val_records, fs_target=fs_target, preprocess_config=preprocess_config
    )
    print(f"   Validation: X={X_val.shape}, classes={np.unique(y_val) if len(y_val) > 0 else 'none'}")
    if len(y_val) > 0:
        val_counts = count_beats_per_class(y_val.tolist())
        print(f"   Class counts: {val_counts}")

    print("\n3. Collecting test beats...")
    X_test, y_test, test_rec_ids, _, _ = collect_beats_from_records(
        loader, test_records, fs_target=fs_target, preprocess_config=preprocess_config
    )
    print(f"   Test: X={X_test.shape}, classes={np.unique(y_test) if len(y_test) > 0 else 'none'}")
    if len(y_test) > 0:
        test_counts = count_beats_per_class(y_test.tolist())
        print(f"   Class counts: {test_counts}")

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError("No beats collected. Check data availability and preprocessing.")

    # Train models
    print("\n4. Training Random Forest...")
    rf_config = ModelConfig(
        model_type="random_forest",
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    rf_model, rf_train_info = train_model(X_train, y_train, rf_config, X_val, y_val)

    print("\n5. Training Logistic Regression...")
    lr_config = ModelConfig(
        model_type="logistic_regression",
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=42
    )
    lr_model, lr_train_info = train_model(X_train, y_train, lr_config, X_val, y_val)

    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    rf_eval = evaluate_model(rf_model, X_test, y_test, class_names=AAMI_CLASSES)
    lr_eval = evaluate_model(lr_model, X_test, y_test, class_names=AAMI_CLASSES)

    print("\n" + "=" * 60)
    print("RANDOM FOREST - TEST RESULTS")
    print("=" * 60)
    from sklearn.metrics import classification_report
    # `labels` MUST be passed alongside `target_names`: sklearn otherwise
    # sorts the observed classes alphabetically (['F','N','S','V']) and maps
    # AAMI_CLASSES (['N','S','V','F']) onto them positionally, printing every
    # class name against the wrong support.
    print(classification_report(y_test, rf_model.predict(X_test),
                                labels=AAMI_CLASSES, target_names=AAMI_CLASSES,
                                zero_division=0))
    print(f"Balanced Accuracy: {rf_eval['balanced_accuracy']:.4f}")
    print(f"Macro F1: {rf_eval['macro_f1']:.4f}")
    print(f"Weighted F1: {rf_eval['weighted_f1']:.4f}")

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION - TEST RESULTS")
    print("=" * 60)
    print(classification_report(y_test, lr_model.predict(X_test),
                                labels=AAMI_CLASSES, target_names=AAMI_CLASSES,
                                zero_division=0))
    print(f"Balanced Accuracy: {lr_eval['balanced_accuracy']:.4f}")
    print(f"Macro F1: {lr_eval['macro_f1']:.4f}")
    print(f"Weighted F1: {lr_eval['weighted_f1']:.4f}")

    # Per-record evaluation
    print("\n7. Per-record evaluation...")
    rf_per_record = evaluate_per_record(rf_model, X_test, y_test, test_rec_ids, AAMI_CLASSES)
    lr_per_record = evaluate_per_record(lr_model, X_test, y_test, test_rec_ids, AAMI_CLASSES)

    # Save everything
    _save_mitbih_results(
        protocol, split, fs_target, preprocess_config,
        rf_model, rf_train_info, rf_eval, rf_per_record,
        lr_model, lr_train_info, lr_eval, lr_per_record,
        train_counts, val_counts if len(y_val) > 0 else {}, test_counts,
        train_records, val_records, test_records, feat_names
    )

    # Generate plots
    _plot_mitbih_results(rf_eval, lr_eval, rf_model, lr_model, X_test, y_test,
                         test_rec_ids, MITBIH_OUT / protocol, feat_names)

    return rf_eval, lr_eval


def _save_mitbih_results(
    protocol, split, fs_target, preprocess_config,
    rf_model, rf_train_info, rf_eval, rf_per_record,
    lr_model, lr_train_info, lr_eval, lr_per_record,
    train_counts, val_counts, test_counts,
    train_records, val_records, test_records, feat_names
):
    """Save all MIT-BIH experiment results."""
    import json

    # The output directory does not exist on a fresh clone / first run.
    out_dir = MITBIH_OUT / protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "mitbih",
        "protocol": protocol,
        "split": {
            "train_records": train_records,
            "val_records": val_records,
            "test_records": test_records,
            "protocol_description": split.description
        },
        "preprocessing": preprocess_config.to_dict(),
        "fs_target": fs_target,
        "feature_names": list(feat_names),
        "class_counts": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts
        },
        "random_forest": {
            "train_info": rf_train_info,
            "test_evaluation": rf_eval,
            "per_record": rf_per_record,
            "feature_importances": get_feature_importance(rf_model, feat_names)
        },
        "logistic_regression": {
            "train_info": lr_train_info,
            "test_evaluation": lr_eval,
            "per_record": lr_per_record,
            "feature_importances": get_feature_importance(lr_model, feat_names)
        }
    }

    # Save models
    save_model(rf_model, MITBIH_OUT / protocol / "rf_model.joblib", {"protocol": protocol, "model_type": "random_forest"})
    save_model(lr_model, MITBIH_OUT / protocol / "lr_model.joblib", {"protocol": protocol, "model_type": "logistic_regression"})

    # Save metrics
    with open(MITBIH_OUT / protocol / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results -> {MITBIH_OUT / protocol}")


def _plot_mitbih_results(rf_eval, lr_eval, rf_model, lr_model, X_test, y_test,
                         test_rec_ids, out_dir, feat_names):
    """Generate plots for MIT-BIH experiment."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, eval_result, title in [(axes[0], rf_eval, "Random Forest"), (axes[1], lr_eval, "Logistic Regression")]:
        cm = np.array(eval_result["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_xticklabels(AAMI_CLASSES)
        ax.set_yticks(range(4)); ax.set_yticklabels(AAMI_CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{title} Confusion Matrix")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Per-class F1 comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(4)
    width = 0.35
    rf_f1 = [rf_eval["per_class"][c]["f1"] for c in AAMI_CLASSES]
    lr_f1 = [lr_eval["per_class"][c]["f1"] for c in AAMI_CLASSES]
    ax.bar(x - width/2, rf_f1, width, label="Random Forest", color="#2563AE")
    ax.bar(x + width/2, lr_f1, width, label="Logistic Regression", color="#1B3A6B")
    ax.set_xticks(x)
    ax.set_xticklabels(AAMI_CLASSES)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 Score Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_f1.png", dpi=150)
    plt.close(fig)

    # Feature importance - Random Forest
    rf_imp = get_feature_importance(rf_model, feat_names)
    if rf_imp:
        fig, ax = plt.subplots(figsize=(8, 4))
        items = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)
        ax.bar([k for k, v in items], [v for k, v in items], color="#2563AE")
        ax.set_ylabel("Importance")
        ax.set_title("Random Forest Feature Importances (MIT-BIH)")
        fig.tight_layout()
        fig.savefig(out_dir / "rf_feature_importance.png", dpi=150)
        plt.close(fig)

    # Feature importance - Logistic Regression
    lr_imp = get_feature_importance(lr_model, feat_names)
    if lr_imp:
        fig, ax = plt.subplots(figsize=(8, 4))
        items = sorted(lr_imp.items(), key=lambda x: x[1], reverse=True)
        ax.bar([k for k, v in items], [v for k, v in items], color="#1B3A6B")
        ax.set_ylabel("Importance (|coef|)")
        ax.set_title("Logistic Regression Feature Importances (MIT-BIH)")
        fig.tight_layout()
        fig.savefig(out_dir / "lr_feature_importance.png", dpi=150)
        plt.close(fig)

    print(f"Saved plots -> {out_dir}")


def run_rpeak_evaluation_on_mitbih(loader, records, protocol_name):
    """Evaluate R-peak detector performance on MIT-BIH records."""
    print(f"\nR-peak detection evaluation on {protocol_name} records...")

    all_scores = []
    for record_name in records:
        try:
            rec_data = loader.load_record(record_name)
        except Exception as e:
            print(f"  Failed to load {record_name}: {e}")
            continue

        signal = rec_data["signal"]
        fs = rec_data["fs"]
        r_peaks_true = rec_data["r_peaks"]
        labels = rec_data["labels"]

        # Filter to AAMI beats for fair comparison
        r_peaks_aami, _, _ = filter_aami_beats(r_peaks_true, labels)

        # Preprocess
        from preprocessing import preprocess_ecg, PreprocessingConfig
        config = PreprocessingConfig(lowcut_hz=0.5, highcut_hz=40.0, filter_order=4)
        signal_proc, _ = preprocess_ecg(signal, fs, config)

        # Detect
        r_det = detect_rpeaks_mitbih(signal_proc, fs)

        # Score
        score = score_detector_detailed(r_det, r_peaks_aami, fs, tolerance_s=0.05)
        score["record"] = record_name
        all_scores.append(score)
        print(f"  {record_name}: sens={score['sensitivity']:.3f}, prec={score['precision']:.3f}, "
              f"f1={score['f1']:.3f}, timing_err={score['mean_timing_error_ms']:.1f}ms")

    if all_scores:
        mean_sens = np.mean([s["sensitivity"] for s in all_scores])
        mean_prec = np.mean([s["precision"] for s in all_scores])
        mean_f1 = np.mean([s["f1"] for s in all_scores])
        mean_timing = np.mean([s["mean_timing_error_ms"] for s in all_scores])
        print(f"  MEAN: sens={mean_sens:.3f}, prec={mean_prec:.3f}, f1={mean_f1:.3f}, timing_err={mean_timing:.1f}ms")

    return all_scores


def run_rpeak_detection_experiment(protocol: str = "classic"):
    """
    Standalone R-peak detector evaluation on real MIT-BIH recordings.

    Kept deliberately separate from beat classification: detector accuracy is
    scored against the reference annotations with one-to-one matching at
    +/-50 ms, and is written to its own metrics file so it can never be
    confused with classification performance.

    Returns
    -------
    dict with pooled aggregate metrics and per-record scores.
    """
    import json

    print("=" * 60)
    print(f"MIT-BIH R-PEAK DETECTION EVALUATION (protocol: {protocol})")
    print("=" * 60)

    if protocol == "classic":
        split = get_classic_split()
    elif protocol == "leakage_audited":
        split = get_leakage_audited_split()
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    loader = MITBIHLoader()
    available = set(loader.get_available_records())
    requested = sorted(set(split.train_records) | set(split.val_records)
                       | set(split.test_records))
    records = [r for r in requested if r in available]
    missing = sorted(set(requested) - available)

    print(f"Records requested: {len(requested)}, available locally: {len(records)}")
    if missing:
        print(f"Missing (skipped): {missing}")

    scores = run_rpeak_evaluation_on_mitbih(loader, records, protocol)

    if not scores:
        raise RuntimeError(
            "No records evaluated. Download the MIT-BIH database first "
            "(see README.md)."
        )

    tp = sum(s["tp"] for s in scores)
    fp = sum(s["fp"] for s in scores)
    fn = sum(s["fn"] for s in scores)
    pooled_sens = tp / (tp + fn) if (tp + fn) else 0.0
    pooled_prec = tp / (tp + fp) if (tp + fp) else 0.0
    pooled_f1 = (2 * pooled_sens * pooled_prec / (pooled_sens + pooled_prec)
                 if (pooled_sens + pooled_prec) > 0 else 0.0)

    results = {
        "experiment": "mitbih_rpeak_detection",
        "protocol": protocol,
        "tolerance_ms": 50,
        "matching": "one-to-one (greedy closest-first)",
        "reference": "MIT-BIH atr annotations restricted to AAMI beat symbols",
        "records_evaluated": records,
        "records_missing": missing,
        "aggregate": {
            "tp": tp, "fp": fp, "fn": fn,
            "pooled_sensitivity": round(pooled_sens, 4),
            "pooled_precision": round(pooled_prec, 4),
            "pooled_f1": round(pooled_f1, 4),
            "mean_per_record_sensitivity": round(float(np.mean(
                [s["sensitivity"] for s in scores])), 4),
            "mean_per_record_precision": round(float(np.mean(
                [s["precision"] for s in scores])), 4),
            "mean_per_record_f1": round(float(np.mean(
                [s["f1"] for s in scores])), 4),
            "mean_timing_error_ms": round(float(np.mean(
                [s["mean_timing_error_ms"] for s in scores])), 2),
            "std_timing_error_ms_across_records": round(float(np.std(
                [s["mean_timing_error_ms"] for s in scores])), 2),
            "n_records": len(scores),
        },
        "per_record": [
            {k: v for k, v in s.items() if k != "matched_pairs"} for s in scores
        ],
    }

    out_dir = MITBIH_OUT / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rpeak_detection.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "-" * 60)
    print("POOLED R-PEAK DETECTION RESULT (real MIT-BIH)")
    print(f"  TP={tp} FP={fp} FN={fn}")
    print(f"  Sensitivity = {pooled_sens:.4f}")
    print(f"  Precision   = {pooled_prec:.4f}")
    print(f"  F1          = {pooled_f1:.4f}")
    print(f"  Mean timing error = {results['aggregate']['mean_timing_error_ms']:.2f} ms")
    print(f"Saved -> {out_dir / 'rpeak_detection.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ECG Arrhythmia Classification Pipeline")
    parser.add_argument("--mode", choices=["synthetic", "mitbih", "rpeaks"],
                        default="synthetic",
                        help="Experiment mode: 'synthetic' runs the original "
                             "synthetic experiment, 'mitbih' runs the real "
                             "AAMI classification experiment, 'rpeaks' runs "
                             "the standalone R-peak detector evaluation")
    parser.add_argument("--protocol", choices=["classic", "leakage_audited"], default="classic",
                        help="MIT-BIH evaluation protocol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic")
    parser.add_argument("--duration", type=float, default=600.0, help="Duration in seconds for synthetic")
    parser.add_argument("--fs-target", type=int, default=360, help="Target sampling frequency for MIT-BIH")

    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic_experiment(seed=args.seed, duration_s=args.duration)
    elif args.mode == "mitbih":
        run_mitbih_experiment(protocol=args.protocol, fs_target=args.fs_target)
    elif args.mode == "rpeaks":
        run_rpeak_detection_experiment(protocol=args.protocol)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()