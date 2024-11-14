import cv2
import numpy as np
import skimage as ski
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from .. import graphic


class MangoFeatureExtractor:
    CHANNELS = ("r", "g", "b")
    STATS = ("mean", "std_dev")
    PCA_COMPONENTS = {"area": 1, "stat": 2, "hist": 32}

    def __init__(self, img):
        self.image = img
        self.label = int(img["label"] == "Ripe")
        self._extract_features()

    def __repr__(self):
        """Return a string representation of the image."""
        return str(self.image)

    def __iter__(self):
        """Yield label and features over the detector's results."""
        yield ("area", self.area)
        yield ("label", self.label)
        yield from self._get_features()

    def _extract_features(self):
        """Extract features from the mango image."""
        img = graphic.load_image("train", self.image)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = self._get_binary_mask(gray_img)
        self.area = np.sum(mask)

        if self.area > 0:
            self._set_features(img, mask)
        else:
            self.mean_r = self.mean_g = self.mean_b = 0.0
            self.std_dev_r = self.std_dev_g = self.std_dev_b = 0.0
            self.histogram = np.zeros(256, dtype=int)

    def _set_features(self, image, mask):
        """Calculate the statistics of the image."""
        for channel, suffix in enumerate(self.CHANNELS):
            pixel_values = image[:, :, channel][mask > 0]

            stats = {"mean": np.mean(pixel_values), "std_dev": np.std(pixel_values)}
            for name, value in stats.items():
                setattr(self, f"{name}_{suffix}", value)

            histogram = np.histogram(pixel_values, bins=256, range=(0, 255))[0]
            setattr(self, f"hist_{suffix}", histogram)

        self.histogram = np.concatenate([self.hist_r, self.hist_g, self.hist_b])

    def _get_features(self):
        """Yield statistical features of the image."""
        for stat in self.STATS:
            for channel in self.CHANNELS:
                yield (f"{stat}_{channel}", getattr(self, f"{stat}_{channel}"))

        for i, value in enumerate(self.histogram):
            yield (f"hist_{i}", value)

    @staticmethod
    def _get_binary_mask(gray_img):
        """Generate a binary mask from the grayscale image using Otsu's thresholding."""
        blurred_img = ski.filters.gaussian(gray_img, sigma=1.0)
        binary_img = blurred_img > threshold_otsu(blurred_img)
        filled_img = binary_fill_holes(binary_img)
        labeled_img, _ = ski.measure.label(filled_img, connectivity=2, return_num=True)
        return remove_small_objects(labeled_img, min_size=100)

    @classmethod
    def get_pca(cls, name, features):
        """Get the principal components of the image features."""
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        pca = PCA(n_components=cls.PCA_COMPONENTS[name])
        pca_features = pca.fit_transform(scaled_features)
        pca_features = pd.DataFrame(pca_features)
        pca_features.columns = [f"{name}_{i}" for i in range(pca_features.shape[1])]
        return pca_features

    @classmethod
    def get_pca_features(cls, features):
        """Get the principal components of the image features."""
        features = pd.DataFrame(features)

        area = features[["area"]]
        X_area = cls.get_pca("area", area)

        stats_id = ["mean_r", "mean_g", "mean_b", "std_dev_r", "std_dev_g", "std_dev_b"]
        stats = features[stats_id]
        X_stats = cls.get_pca("stat", stats)

        histograms = features[[f"hist_{i}" for i in range(768)]]
        X_histogram = cls.get_pca("hist", histograms)

        return pd.concat([X_area, X_stats, X_histogram], axis=1)
