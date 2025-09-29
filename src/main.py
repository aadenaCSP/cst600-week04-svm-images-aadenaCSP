from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
import joblib

from .train_eval import (
    evaluate_and_plot,
    load_folder_dataset,
    make_pipeline,
    run_grid_search,
    save_cv_results,
    summarize_class_counts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVM Image Classification (HOG baseline)")
    p.add_argument("--data_dir", type=str, default="data/raw", help="Root folder with class subdirs")

    # Preprocessing / features
    p.add_argument("--img_size", type=int, nargs=2, metavar=("H", "W"), default=(128, 128))
    p.add_argument("--grayscale", action="store_true", help="Use grayscale for HOG/LBP")
    p.add_argument("--include_color_hist", action="store_true", help="Append per-channel color histograms")
    p.add_argument("--hist_bins", type=int, default=16)
    p.add_argument("--include_lbp", action="store_true", help="Append LBP histogram")
    p.add_argument("--lbp_points", type=int, default=24)
    p.add_argument("--lbp_radius", type=int, default=3)

    # HOG params
    p.add_argument("--hog_orientations", type=int, default=9)
    p.add_argument("--hog_ppc", type=int, nargs=2, metavar=("PH", "PW"), default=(8, 8), help="HOG pixels_per_cell")
    p.add_argument("--hog_cpb", type=int, nargs=2, metavar=("CH", "CW"), default=(2, 2), help="HOG cells_per_block")
    p.add_argument("--hog_block_norm", type=str, default="L2-Hys")
    p.add_argument("--hog_transform_sqrt", action="store_true")

    # SVM / training
    p.add_argument("--probabilities", action="store_true", help="Enable probability=True on SVC (slower)")
    p.add_argument("--class_weight", type=str, default=None, choices=[None, "balanced"], help="Class weighting for SVC")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--cv_splits", type=int, default=5)
    p.add_argument("--scoring", type=str, default="f1_macro")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--max_per_class", type=int, default=None, help="Cap images per class (optional)")

    # Output paths
    p.add_argument("--outputs_dir", type=str, default="outputs")
    p.add_argument("--figures_dir", type=str, default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)
    figures_dir = Path(args.figures_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset (paths + labels)
    X_paths, y_labels, class_names = load_folder_dataset(
        data_dir=data_dir, max_per_class=args.max_per_class
    )
    print("Loaded images:", len(X_paths))
    print(summarize_class_counts(y_labels))

    # 2) Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_paths, y_labels, test_size=args.test_size, random_state=args.random_state, stratify=y_labels
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Classes: {class_names}")

    # 3) Build pipeline (features -> scaler -> SVC)
    pipe = make_pipeline(
        image_size=tuple(args.img_size),
        grayscale=args.grayscale,
        include_lbp=args.include_lbp,
        lbp_points=args.lbp_points,
        lbp_radius=args.lbp_radius,
        include_color_hist=args.include_color_hist,
        hist_bins=args.hist_bins,
        hog_orientations=args.hog_orientations,
        hog_pixels_per_cell=tuple(args.hog_ppc),
        hog_cells_per_block=tuple(args.hog_cpb),
        hog_block_norm=args.hog_block_norm,
        hog_transform_sqrt=args.hog_transform_sqrt,
        probability=args.probabilities,
        class_weight=args.class_weight,
    )

    # 4) Grid search (CV on training set)
    grid = run_grid_search(
        pipe,
        X_train,
        y_train,
        scoring=args.scoring,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    # Save CV table
    save_cv_results(grid, outputs_dir / "cv_results.csv")

    print("\nBest CV score (", args.scoring, "):", grid.best_score_)
    print("Best params:", grid.best_params_)

    # 5) Evaluate on held-out test set
    best_model = grid.best_estimator_
    metrics_dict = evaluate_and_plot(
        best_model,
        X_test,
        y_test,
        class_names=class_names,
        out_dir=str(figures_dir),
        report_path=str(outputs_dir / "test_report.txt"),
    )

    print("\nTest Accuracy:", f"{metrics_dict['accuracy']:.4f}")
    print("Test AUC (macro OVR):", metrics_dict["auc_macro_ovr"])  # may be nan if not computable

    # 6) Persist best model
    model_path = outputs_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print("Saved best model to:", model_path)


if __name__ == "__main__":
    main()
