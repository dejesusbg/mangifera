import cv2
import numpy as np
import skimage as ski
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from . import graphic


class MangoFeatureExtractor:
    CHANNELS = ("r", "g", "b")
    STATS = ("mean", "std_dev")

    def __init__(self, image):
        self.image = image
        self.mean = {}
        self.std_dev = {}
        self.histogram = np.zeros(256, dtype=int)
        self._extract_features()

    def __repr__(self):
        """Return a string representation of the image."""
        return str(self.image)

    def __iter__(self):
        """Yield label and features over the detector's results."""
        yield ("area", self.area)
        yield from self._get_features()

    def _extract_features(self):
        """Extract features from the mango image."""
        img = graphic.load_image("train", self.image)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = self._get_binary_mask(gray_img)
        self.area = np.sum(mask)

        if self.area > 0:
            self._set_features(img, mask)

    def _set_features(self, image, mask):
        """Calculate the statistics of the image."""
        histograms = []
        for idx, channel in enumerate(self.CHANNELS):
            pixel_values = image[:, :, idx][mask > 0]

            self.mean[channel] = np.mean(pixel_values)
            self.std_dev[channel] = np.std(pixel_values)

            hist = np.histogram(pixel_values, bins=256, range=(0, 255))[0]
            histograms.append(hist)

        self.histogram = np.concatenate(histograms)

    def _get_features(self):
        """Yield statistical features of the image."""
        for stat in self.STATS:
            for channel in self.CHANNELS:
                yield (f"{stat}_{channel}", getattr(self, stat)[channel])

        for i, value in enumerate(self.histogram):
            yield (f"hist_{i}", value)

    @staticmethod
    def _get_binary_mask(gray_img):
        """Generate a binary mask from the grayscale image using Otsu's thresholding."""
        blurred_img = ski.filters.gaussian(gray_img, sigma=1.0)
        binary_mask = blurred_img > threshold_otsu(blurred_img)
        filled_mask = binary_fill_holes(binary_mask)
        labeled_img, _ = ski.measure.label(filled_mask, connectivity=2, return_num=True)
        return remove_small_objects(labeled_img, min_size=100)

    @staticmethod
    def _get_pca(features, n_components):
        """Get the principal components of the image features."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        pca = PCA(n_components=n_components)
        return pca.fit_transform(scaled_features)

    @classmethod
    def get_pca_features(cls, features, hist_components):
        """Get the principal components of the image features."""
        features = pd.DataFrame(features)

        area = features[["area"]]
        X_area = cls._get_pca(area, 1)

        stats_id = ["mean_r", "mean_g", "mean_b", "std_dev_r", "std_dev_g", "std_dev_b"]
        stats = features[stats_id]
        X_stats = cls._get_pca(stats, 2)

        histograms = features[[f"hist_{i}" for i in range(768)]]
        X_histogram = cls._get_pca(histograms, hist_components)

        return np.concatenate([X_area, X_stats, X_histogram], axis=1)
