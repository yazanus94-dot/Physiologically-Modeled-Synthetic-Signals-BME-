"""
Model training module for ECG arrhythmia classification.

Supports Random Forest and Logistic Regression baselines with
proper hyperparameter configuration and reproducibility.
"""
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    f1_score, precision_recall_fscore_support
)


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str = "random_forest"  # "random_forest", "logistic_regression"
    random_state: int = 42

    # Random Forest params
    n_estimators: int = 200
    max_depth: Optional[int] = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "balanced"
    n_jobs: int = -1

    # Logistic Regression params
    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000

    # Cross-validation
    cv_folds: int = 5
    use_cv: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**d)


def create_model(config: ModelConfig):
    """Create model from configuration."""
    if config.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            class_weight=config.class_weight,
            random_state=config.random_state,
            n_jobs=config.n_jobs
        )
    elif config.model_type == "logistic_regression":
        # Logistic Regression needs feature scaling
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=config.C,
                penalty=config.penalty,
                solver=config.solver,
                max_iter=config.max_iter,
                class_weight=config.class_weight if hasattr(config, 'class_weight') else "balanced",
                random_state=config.random_state,
                n_jobs=config.n_jobs if hasattr(config, 'n_jobs') else -1
            ))
        ])
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ModelConfig,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None
):
    """
    Train model with given configuration.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    config : ModelConfig
        Model configuration
    X_val, y_val : np.ndarray, optional
        Validation data (for early stopping / model selection)
    groups : array-like, optional
        One group label per training sample — the record (patient) each
        beat came from.  Required whenever ``config.use_cv`` is set:
        cross-validation must hold out whole records, never random beats.

    Returns
    -------
    model : fitted estimator
    train_info : dict
        Training information
    """
    model = create_model(config)

    train_info = {
        "config": config.to_dict(),
        "n_train_samples": len(X_train),
        "n_features": X_train.shape[1],
        "classes": sorted(np.unique(y_train).tolist()),
        "class_counts": {c: int(np.sum(y_train == c)) for c in np.unique(y_train)}
    }

    if config.use_cv:
        # RECORD-GROUPED cross-validation.  Beats from one record are
        # highly correlated (same patient, same session, same electrode
        # placement), so a beat-level split that shuffles them across
        # folds leaks patient identity into the validation fold and
        # inflates the score.  GroupKFold keeps every beat of a record in
        # exactly one fold, so each validation fold is a set of patients
        # the model has not seen.
        if groups is None:
            raise ValueError(
                "config.use_cv=True requires `groups` (the record id of "
                "each training sample). Record-grouped validation is "
                "mandatory here: beat-level cross-validation leaks "
                "patient identity across folds.")
        groups = np.asarray(groups)
        if groups.shape[0] != np.asarray(X_train).shape[0]:
            raise ValueError(
                "groups length %d != n_train_samples %d"
                % (groups.shape[0], np.asarray(X_train).shape[0]))

        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError(
                "record-grouped CV needs >= 2 distinct records, got %d"
                % n_groups)
        n_splits = int(min(config.cv_folds, n_groups))
        cv = GroupKFold(n_splits=n_splits)
        train_info["cv_scheme"] = "record-grouped (GroupKFold)"
        train_info["cv_folds_used"] = n_splits
        train_info["cv_n_groups"] = int(n_groups)

        if X_val is not None and len(np.asarray(X_val)) > 0:
            # Keep the caller's held-out validation beats out of the CV
            # pool entirely — they are a separate patient-level split.
            X_fit = np.asarray(X_train)
            y_fit = np.asarray(y_train)
            g_fit = groups
        else:
            X_fit = np.asarray(X_train)
            y_fit = np.asarray(y_train)
            g_fit = groups

        # For CV, we need to define param grid based on model type
        if config.model_type == "random_forest":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10]
            }
        else:
            param_grid = {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l2"]
            }

        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring="f1_macro",
            n_jobs=config.n_jobs, verbose=1
        )
        grid_search.fit(X_fit, y_fit, groups=g_fit)

        model = grid_search.best_estimator_
        train_info["cv_best_params"] = grid_search.best_params_
        train_info["cv_best_score"] = float(grid_search.best_score_)

        # Refit on everything the caller handed over (train only; the
        # separate validation split stays untouched).
        model.fit(np.asarray(X_train), np.asarray(y_train))
    else:
        model.fit(X_train, y_train)

    return model, train_info


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.

    Returns
    -------
    dict with keys:
        - classification_report (dict)
        - confusion_matrix (np.ndarray)
        - balanced_accuracy (float)
        - macro_f1 (float)
        - weighted_f1 (float)
        - per_class_precision (dict)
        - per_class_recall (dict)
        - per_class_f1 (dict)
        - per_class_support (dict)
    """
    y_pred = model.predict(X_test)

    if class_names is None:
        class_names = sorted(set(np.unique(y_test).tolist())
                             | set(np.unique(y_pred).tolist()))

    # `target_names` is paired POSITIONALLY with `labels`, and sklearn
    # otherwise infers `labels` by sorting the observed classes. With
    # class_names = ['N','S','V','F'] but observed labels sorted as
    # ['F','N','S','V'], every printed class name would silently shift by one
    # position. Passing `labels` explicitly makes the pairing deterministic
    # and also tolerates classes that are absent from a (per-record) subset.
    report = classification_report(
        y_test, y_pred,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    # Overall metrics
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=class_names, zero_division=0
    )

    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "class_names": class_names
    }


def evaluate_per_record(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    record_ids: List[str],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance per record.

    Returns per-record metrics for detailed analysis.
    """
    if class_names is None:
        # Passed through to evaluate_model(), which resolves the label set
        # itself; only used here for per-record class_counts.
        class_names = sorted(np.unique(y_test).tolist())

    unique_records = np.unique(record_ids)
    per_record = {}

    for record in unique_records:
        mask = np.array([r == record for r in record_ids])
        if mask.sum() == 0:
            continue

        X_rec = X_test[mask]
        y_rec = y_test[mask]

        if len(np.unique(y_rec)) < 2:
            continue

        eval_result = evaluate_model(model, X_rec, y_rec, class_names)
        per_record[record] = {
            "n_beats": int(mask.sum()),
            "class_counts": {c: int(np.sum(y_rec == c)) for c in class_names},
            "balanced_accuracy": eval_result["balanced_accuracy"],
            "macro_f1": eval_result["macro_f1"],
            "weighted_f1": eval_result["weighted_f1"],
            "per_class": eval_result["per_class"]
        }

    return {"per_record": per_record}


def get_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importance from trained model."""
    if hasattr(model, "feature_importances_"):
        # Random Forest
        importances = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "coef_"):
        # Logistic Regression in pipeline
        importances = np.abs(model.named_steps["clf"].coef_).mean(axis=0)
    else:
        return {}

    return dict(zip(feature_names, [float(v) for v in importances]))


def save_model(model, path: str, metadata: Optional[Dict] = None):
    """Save model with metadata."""
    save_dict = {"model": model}
    if metadata:
        save_dict["metadata"] = metadata
    joblib.dump(save_dict, path)


def load_model(path: str):
    """Load model and metadata."""
    data = joblib.load(path)
    if isinstance(data, dict) and "model" in data:
        return data["model"], data.get("metadata", {})
    return data, {}


if __name__ == "__main__":
    # Quick test with synthetic data
    from generate_signal import generate_ecg
    from detect_rpeaks import detect_rpeaks
    from features import extract_features
    from sklearn.model_selection import train_test_split

    signal, r_true, labels, fs = generate_ecg(duration_s=60, fs=250, seed=1)
    r_det = detect_rpeaks(signal, fs)
    X, feat_names = extract_features(signal, r_det, fs)

    # Align labels
    tol = int(0.05 * fs)
    y = []
    keep = []
    for i, p in enumerate(r_det):
        if i == 0 or i == len(r_det) - 1:
            continue
        j = np.argmin(np.abs(r_true - p))
        if abs(p - r_true[j]) <= tol:
            y.append(labels[j])
            keep.append(i - 1)  # features.py skips first/last

    X = X[keep]
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Train RF
    config = ModelConfig(model_type="random_forest", n_estimators=100)
    model, info = train_model(X_train, y_train, config)
    print("Train info:", info)

    eval_result = evaluate_model(model, X_test, y_test, class_names=["N", "V", "S"])
    print("Evaluation:", eval_result)

    # Train LR
    config_lr = ModelConfig(model_type="logistic_regression", C=1.0)
    model_lr, info_lr = train_model(X_train, y_train, config_lr)
    eval_lr = evaluate_model(model_lr, X_test, y_test, class_names=["N", "V", "S"])
    print("LR Evaluation:", eval_lr)