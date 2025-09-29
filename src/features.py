from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
from skimage import io, color, transform
from skimage.feature import hog, local_binary_pattern
from sklearn.base import BaseEstimator, TransformerMixin


def _is_image_file(path: str) -> bool:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return str(path).lower().endswith(exts)


class ImageToFeatures(BaseEstimator, TransformerMixin):
    """
    Convert image file paths to feature vectors (HOG ± LBP ± color hist).
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        grayscale: bool = True,
        hog_orientations: int = 9,
        hog_pixels_per_cell: Tuple[int, int] = (8, 8),
        hog_cells_per_block: Tuple[int, int] = (2, 2),
        hog_block_norm: str = "L2-Hys",
        hog_transform_sqrt: bool = True,
        include_lbp: bool = False,
        lbp_points: int = 24,
        lbp_radius: int = 3,
        include_color_hist: bool = False,
        hist_bins: int = 16,
        normalize_hist: bool = True,
    ) -> None:
        self.image_size = image_size
        self.grayscale = grayscale
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.hog_block_norm = hog_block_norm
        self.hog_transform_sqrt = hog_transform_sqrt
        self.include_lbp = include_lbp
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.include_color_hist = include_color_hist
        self.hist_bins = hist_bins
        self.normalize_hist = normalize_hist

    def fit(self, X: Sequence[str], y: Sequence[int] | None = None):
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        feats: List[np.ndarray] = []
        for path in X:
            if not _is_image_file(path):
                raise ValueError(f"Not an image file: {path}")
            img = io.imread(path)

            # Ensure float32 in [0,1]
            if img.dtype.kind in {"u", "i"}:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)

            # Resize
            img_resized = transform.resize(
                img, self.image_size, anti_aliasing=True, preserve_range=True
            ).astype(np.float32)

            # Grayscale for HOG/LBP
            if img_resized.ndim == 3:
                gray = color.rgb2gray(img_resized)
            else:
                gray = img_resized

            # HOG
            hog_vec = hog(
                gray,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                block_norm=self.hog_block_norm,
                transform_sqrt=self.hog_transform_sqrt,
                feature_vector=True,
            ).astype(np.float32)

            parts = [hog_vec]

            # LBP histogram
            if self.include_lbp:
                lbp = local_binary_pattern(
                    gray, P=self.lbp_points, R=self.lbp_radius, method="uniform"
                )
                n_bins = self.lbp_points + 2
                lbp_hist, _ = np.histogram(
                    lbp.ravel(), bins=n_bins, range=(0, n_bins), density=self.normalize_hist
                )
                parts.append(lbp_hist.astype(np.float32))

            # Color hist per channel (if requested)
            if self.include_color_hist:
                if img_resized.ndim == 2:
                    rgb = color.gray2rgb(img_resized)
                else:
                    rgb = img_resized
                ch_hists = []
                for ch in range(rgb.shape[2]):
                    hist, _ = np.histogram(
                        rgb[..., ch].ravel(),
                        bins=self.hist_bins,
                        range=(0.0, 1.0),
                        density=self.normalize_hist,
                    )
                    ch_hists.append(hist.astype(np.float32))
                parts.append(np.concatenate(ch_hists))

            feats.append(np.concatenate(parts))

        return np.vstack(feats)
