import cv2
import numpy as np
import skimage as ski
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from . import graphic


class MangoFeatureExtractor:
    CHANNELS = ("r", "g", "b")
    STATS = ("mean", "std_dev")

    def __init__(self, split, image):
        self.split = split
        self.image = image
        self.mean = {}
        self.std_dev = {}
        self.histogram = np.zeros(256, dtype=int)
        self._extract_features()

    def __iter__(self):
        """Yield the area and the statistical features of the image."""
        yield ("area", self.area)
        yield from self._get_features()

    def _extract_features(self):
        """Extract relevant features from the mango image."""
        img = graphic.load_image(self.split, self.image)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = self._get_binary_mask(gray_img)
        self.area = np.sum(mask)

        if self.area > 0:
            self._set_features(img, mask)

    def _set_features(self, image, mask):
        """Calculate statistical features and histogram of pixel values."""
        histograms = []
        for idx, channel in enumerate(self.CHANNELS):
            pixel_values = image[:, :, idx][mask > 0]

            self.mean[channel] = np.mean(pixel_values)
            self.std_dev[channel] = np.std(pixel_values)

            hist = np.histogram(pixel_values, bins=256, range=(0, 255))[0]
            histograms.append(hist)

        self.histogram = np.concatenate(histograms)

    def _get_features(self):
        """Yield statistical features and histogram values of the image."""
        for stat in self.STATS:
            for channel in self.CHANNELS:
                yield (f"{stat}_{channel}", getattr(self, stat)[channel])

        for i, value in enumerate(self.histogram):
            yield (f"hist_{i}", value)

    @staticmethod
    def _get_binary_mask(gray_img):
        """Generate a binary mask from the grayscale image using Otsu's thresholding method."""
        blurred_img = ski.filters.gaussian(gray_img, sigma=1.0)
        binary_mask = blurred_img > threshold_otsu(blurred_img)
        filled_mask = binary_fill_holes(binary_mask)
        labeled_img, _ = ski.measure.label(filled_mask, connectivity=2, return_num=True)
        return remove_small_objects(labeled_img, min_size=100)

    @staticmethod
    def apply_pca(features, n_components):
        """Apply PCA to the given features to reduce dimensionality."""
        features = pd.DataFrame(features)
        pca = PCA(n_components=n_components)
        return pca.fit_transform(features)

    @staticmethod
    def get_scaled_features(features):
        """Scale the features to a range of 0 to 1."""
        features = pd.DataFrame(features)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
        return scaled_df.to_dict(orient="records")

    @staticmethod
    def get_encoded_labels(data):
        """Encode the labels as integers."""
        le = LabelEncoder()
        labels = pd.DataFrame(data)[["label"]]
        return le.fit_transform(labels)
