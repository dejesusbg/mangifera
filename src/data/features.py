import cv2
import numpy as np
import skimage as ski
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from .. import graphic


class MangoFeatureExtractor:
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
        yield from self._get_stats()
        yield from self._get_histogram()

    def _extract_features(self):
        """Extract features from the mango image."""
        img = self._load_image("train", self.image)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = self._get_binary_mask(gray_img)
        self.area = np.sum(mask)

        if self.area > 0:
            self._set_stats(img, mask, "r", 0)
            self._set_stats(img, mask, "g", 1)
            self._set_stats(img, mask, "b", 2)

            pixel_values = gray_img[mask > 0]
            self._set_histogram(pixel_values)
        else:
            self.mean_gray = self.max_gray = self.min_gray = self.std_dev_gray = 0.0
            self.mean_r = self.max_r = self.min_r = self.std_dev_r = 0.0
            self.mean_g = self.max_g = self.min_g = self.std_dev_g = 0.0
            self.mean_b = self.max_b = self.min_b = self.std_dev_b = 0.0
            self.histogram = np.zeros(256, dtype=int)

    def _set_stats(self, image, mask, suffix, channel):
        """Calculate the statistics of the image."""
        pixel_values = image[:, :, channel][mask > 0]
        stats = {"mean": np.mean(pixel_values), "std_dev": np.std(pixel_values)}

        for name, value in stats.items():
            setattr(self, f"{name}_{suffix}", value)

    def _get_stats(self):
        """Yield statistical features of the image."""
        stats = ("mean", "std_dev")
        channels = ("r", "g", "b")

        for stat in stats:
            for channel in channels:
                yield (f"{stat}_{channel}", getattr(self, f"{stat}_{channel}"))

    def _set_histogram(self, pixel_values):
        """Calculate the histogram of the image."""
        self.histogram = np.histogram(pixel_values, bins=256, range=(0, 255))[0]

    def _get_histogram(self):
        """Yield histogram features of the image."""
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

    @staticmethod
    def _load_image(split, img):
        """Load and resize the image from the specified split."""
        img = graphic.load_image(split, img)
        return np.array(img)
