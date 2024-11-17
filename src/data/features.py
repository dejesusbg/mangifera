import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix as matrix, graycoprops as props
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from . import graphic


class MangoFeatureExtractor:
    CHANNELS = ("r", "g", "b")
    STATS = ("mean", "std_dev")
    TEXTURE = ("contrast", "correlation", "energy", "homogeneity")

    def __init__(self, split, image):
        self.split = split
        self.image = image
        self.mean = {}
        self.std_dev = {}
        self.texture = {}
        self.histogram = np.zeros(256, dtype=int)
        self._extract_features()

    def __iter__(self):
        """Yield the area, statistical features, histogram, and texture features of the image."""
        yield from self._get_features()

    def _extract_features(self):
        """Extract relevant features from the mango image."""
        img = graphic.load_image(self.split, self.image)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.area = gray_img.size
        self._set_features(img)
        self._set_textures(gray_img)

    def _set_features(self, image):
        """Calculate statistical features and histogram of pixel values."""
        histograms = []
        for idx, channel in enumerate(self.CHANNELS):
            pixel_values = image[:, :, idx].flatten()

            self.mean[channel] = np.mean(pixel_values)
            self.std_dev[channel] = np.std(pixel_values)

            hist = np.histogram(pixel_values, bins=256, range=(0, 255))[0]
            histograms.append(hist)

        self.histogram = np.concatenate(histograms)

    def _set_textures(self, gray_img):
        """Compute Haralick texture features using the GLCM."""
        glcm = matrix(gray_img, distances=[1], angles=[0], symmetric=True, normed=True)
        for prop in self.TEXTURE:
            self.texture[prop] = props(glcm, prop)[0, 0]

    def _get_features(self):
        """Yield statistical features, histogram values, and Haralick features of the image."""
        for stat in self.STATS:
            for channel in self.CHANNELS:
                yield (f"{stat}_{channel}", getattr(self, stat)[channel])

        for i, value in enumerate(self.histogram):
            yield (f"hist_{i}", value)

        for prop in self.TEXTURE:
            yield (f"texture_{prop}", self.texture[prop])

    @staticmethod
    def apply_pca(features, n_components):
        """Apply PCA to the given features to reduce dimensionality."""
        features = pd.DataFrame(features)
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features)
        return pca_features, pca.n_components_

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
