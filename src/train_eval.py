from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

from .features import ImageToFeatures


def load_folder_dataset(
    data_dir: str | Path,
    max_per_class: int | None = None,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> Tuple[List[str], List[str], List[str]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    X_paths: List[str] = []
    y_labels: List[str] = []

    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(
            f"No class subfolders found under {data_dir}. Expected data/raw/<class>/*.png etc."
        )

    for cls in classes:
        cls_dir = data_dir / cls
        files = [str(p) for p in cls_dir.rglob("*") if p.suffix.lower() in exts]
        files.sort()
        if max_per_class is not None and len(files) > max_per_class:
            files = files[:max_per_class]
        X_paths.extend(files)
        y_labels.extend([cls] * len(files))

    if len(X_paths) == 0:
        raise ValueError(f"No images found in {data_dir} with extensions {exts}")

    return X_paths, y_labels, classes


def make_pipeline(
    *,
    image_size=(128, 128),
    grayscale=True,
    include_lbp=False,
    lbp_points=24,
    lbp_radius=3,
    include_color_hist=False,
    hist_bins=16,
    hog_orientations=9,
    hog_pixels_per_cell=(8, 8),
    hog_cells_per_block=(2, 2),
    hog_block_norm="L2-Hys",
    hog_transform_sqrt=True,
    probability=False,
    class_weight: str | None = None,
) -> Pipeline:
    feature_transform = ImageToFeatures(
        image_size=image_size,
        grayscale=grayscale,
        hog_orientations=hog_orientations,
        hog_pixels_per_cell=hog_pixels_per_cell,
        hog_cells_per_block=hog_cells_per_block,
        hog_block_norm=hog_block_norm,
        hog_transform_sqrt=hog_transform_sqrt,
        include_lbp=include_lbp,
        lbp_points=lbp_points,
        lbp_radius=lbp_radius,
        include_color_hist=include_color_hist,
        hist_bins=hist_bins,
    )

    svc = SVC(probability=probability, class_weight=class_weight)

    pipe = Pipeline(
        steps=[
            ("features", feature_transform),
            ("scaler", StandardScaler()),
            ("svc", svc),
        ]
    )
    return pipe


def grid_for_svm() -> List[dict]:
    return [
        {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10]},
        {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10], "svc__gamma": ["scale", "auto", 0.01, 0.1]},
        {
            "svc__kernel": ["poly"],
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", "auto", 0.01, 0.1],
            "svc__degree": [2, 3],
        },
    ]


def run_grid_search(
    pipe: Pipeline,
    X_train: Sequence[str],
    y_train: Sequence[int] | Sequence[str],
    *,
    scoring: str = "f1_macro",
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=grid_for_svm(),
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)
    return grid


def save_cv_results(grid: GridSearchCV, out_csv: str | Path) -> None:
    import pandas as pd

    df = pd.DataFrame(grid.cv_results_)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def evaluate_and_plot(
    model: Pipeline,
    X_test: Sequence[str],
    y_test: Sequence[int] | Sequence[str],
    class_names: List[str],
    out_dir: str | Path = "figures",
    report_path: str | Path = "outputs/test_report.txt",
) -> Dict[str, float]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    # ----- Text report -----
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    acc = metrics.accuracy_score(y_test, y_pred)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Test Metrics\n")
        f.write("============\n\n")
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")

    # ----- Confusion matrix -----
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax_cm, values_format='d', colorbar=False)
    plt.title("Confusion Matrix — Test")
    fig_cm.tight_layout()
    fig_cm.savefig(Path(out_dir) / "confusion_matrix.png", dpi=200)
    plt.close(fig_cm)

    # ---------- ROC-AUC & ROC plot ----------
    # Binary: plot single ROC using positive class scores
    # Multiclass: compute macro AUC (OvR) and plot one curve per class.

    # Get raw scores
    scores = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        try:
            scores = model.predict_proba(X_test)
        except Exception:
            scores = None

    auc_macro = float("nan")
    roc_path = Path(out_dir) / "roc_curve.png"

    if scores is not None:
        n_classes = len(class_names)

        if n_classes == 2:
            # Binary case: 1D y and positive-class scores
            pos_class = class_names[1]  # treat the second label as the positive class
            y_true_bin = (np.array(y_test) == pos_class).astype(int)

            # Extract positive class scores
            if scores.ndim == 2 and scores.shape[1] == 2:
                pos_scores = scores[:, 1]
            else:
                # decision_function may return shape (n_samples,) in binary
                pos_scores = scores.ravel()

            try:
                auc_macro = roc_auc_score(y_true_bin, pos_scores)
            except Exception:
                auc_macro = float("nan")

            fig, ax = plt.subplots(figsize=(7, 6))
            RocCurveDisplay.from_predictions(y_true_bin, pos_scores, ax=ax)
            ax.set_title(f"ROC Curve — AUC={auc_macro:.3f}")
            fig.tight_layout()
            fig.savefig(roc_path, dpi=200)
            plt.close(fig)

        else:
            # Multiclass: one-vs-rest per class
            le = LabelEncoder().fit(class_names)
            y_int = le.transform(np.array(y_test))
            y_bin = label_binarize(y_int, classes=np.arange(n_classes))

            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)

            try:
                auc_macro = roc_auc_score(y_bin, scores, average="macro", multi_class="ovr")
            except Exception:
                auc_macro = float("nan")

            fig, ax = plt.subplots(figsize=(7, 6))
            for i, cls in enumerate(class_names):
                try:
                    RocCurveDisplay.from_predictions(y_bin[:, i], scores[:, i], ax=ax, name=str(cls))
                except Exception:
                    continue
            ax.set_title(f"ROC Curves (OvR) — macro AUC={auc_macro:.3f}")
            fig.tight_layout()
            fig.savefig(roc_path, dpi=200)
            plt.close(fig)

    return {"accuracy": acc, "auc_macro_ovr": auc_macro}


def summarize_class_counts(labels: Sequence[str]) -> str:
    from collections import Counter

    counts = Counter(labels)
    lines = ["Class counts:"]
    for k in sorted(counts):
        lines.append(f"  {k}: {counts[k]}")
    return "\n".join(lines)
