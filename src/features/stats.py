import cv2
import numpy as np
import skimage as ski
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from src import Mango


class StatsMango(Mango):
    def _extract_features(self):
        """Extract various features from the mango image."""
        img = self._load_original("train", self.image)
        img_resized = cv2.resize(np.array(img), self.SIZE)
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Create a binary mask for the mango
        mango_mask = self._get_binary_mango(gray_img)

        # Calculate properties with checks for valid mask
        area = np.sum(mango_mask)

        if area > 0:
            mean_gray = np.mean(gray_img[mango_mask > 0])
            max_gray = np.max(gray_img[mango_mask > 0])
            min_gray = np.min(gray_img[mango_mask > 0])
            std_dev_gray = np.std(gray_img[mango_mask > 0])
        else:
            mean_gray = max_gray = min_gray = std_dev_gray = 0.0

        # Mean intensities for each color channel
        mean_intensity_R = (
            np.mean(img_resized[:, :, 0][mango_mask > 0]) if area > 0 else 0.0
        )
        mean_intensity_G = (
            np.mean(img_resized[:, :, 1][mango_mask > 0]) if area > 0 else 0.0
        )
        mean_intensity_B = (
            np.mean(img_resized[:, :, 2][mango_mask > 0]) if area > 0 else 0.0
        )

        # Histogram
        histogram = self._calculate_histogram(area, gray_img[mango_mask > 0])

        # Combine all features into a single dictionary
        return {
            "area": area,
            "mean_gray": mean_gray,
            "max_gray": max_gray,
            "min_gray": min_gray,
            "std_dev_gray": std_dev_gray,
            "mean_intensity_R": mean_intensity_R,
            "mean_intensity_G": mean_intensity_G,
            "mean_intensity_B": mean_intensity_B,
            **histogram,
        }

    def _get_binary_mango(self, gray_image):
        """Get a binary mask of the mango using Otsu's thresholding."""
        blurred_image = ski.filters.gaussian(gray_image, sigma=1.0)
        binary_image = blurred_image > threshold_otsu(blurred_image)
        filled_img = binary_fill_holes(binary_image)
        labeled_image, _ = ski.measure.label(
            filled_img, connectivity=2, return_num=True
        )
        mango_mask = remove_small_objects(labeled_image, min_size=100)
        return mango_mask

    def _calculate_histogram(self, area, pixel_values):
        """Calculate histogram counts for pixel values from 0 to 255."""
        if area <= 0:
            return {f"hist_{i}": 0 for i in range(256)}
        return {f"hist_{i}": np.count_nonzero(pixel_values == i) for i in range(256)}

    def __iter__(self):
        """Yield label and features for iteration."""
        yield from super().__iter__()
        for key, value in self.features.items():
            yield (key, value)

    def __str__(self):
        """Return a string representation of the image."""
        return str(self.image)
