import cv2
import numpy as np
import skimage as ski
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from .. import mango


class MangoStatistics(mango):
    def _extract_features(self):
        """Extract statistical features from the mango image."""
        img = self._load_image("train", self.image)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = self._get_binary_mask(gray_img)
        self.area = np.sum(mask)

        if self.area > 0:
            pixel_values = gray_img[mask > 0]

            self.mean_gray = np.mean(pixel_values)
            self.max_gray = np.max(pixel_values)
            self.min_gray = np.min(pixel_values)
            self.std_dev_gray = np.std(pixel_values)

            mean_intensity = lambda channel: np.mean(img[:, :, channel][mask > 0])
            self.mean_intensity_R = mean_intensity(0)
            self.mean_intensity_G = mean_intensity(1)
            self.mean_intensity_B = mean_intensity(2)

            self.histogram = self._get_histogram(pixel_values)
        else:
            self.mean_gray = self.max_gray = self.min_gray = self.std_dev_gray = 0.0
            self.mean_intensity_R = self.mean_intensity_G = self.mean_intensity_B = 0.0
            self.histogram = np.zeros(256, dtype=int)

    @staticmethod
    def _get_histogram(pixel_values):
        """Calculate the histogram of pixel values."""
        return np.histogram(pixel_values, bins=256, range=(0, 255))[0]

    @staticmethod
    def _get_binary_mask(gray_img):
        """Generate a binary mask from the grayscale image using Otsu's thresholding."""
        blurred_img = ski.filters.gaussian(gray_img, sigma=1.0)
        binary_img = blurred_img > threshold_otsu(blurred_img)
        filled_img = binary_fill_holes(binary_img)
        labeled_img, _ = ski.measure.label(filled_img, connectivity=2, return_num=True)
        return remove_small_objects(labeled_img, min_size=100)

    def __iter__(self):
        """Yield statistical features for iteration over the detector's results."""
        yield from super().__iter__()
        yield ("area", self.area)
        yield ("mean_gray", self.mean_gray)
        yield ("max_gray", self.max_gray)
        yield ("min_gray", self.min_gray)
        yield ("std_dev_gray", self.std_dev_gray)
        yield ("mean_intensity_R", self.mean_intensity_R)
        yield ("mean_intensity_G", self.mean_intensity_G)
        yield ("mean_intensity_B", self.mean_intensity_B)
        yield from self._map_features("hist", self.histogram)
