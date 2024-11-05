import cv2
import numpy as np
from .. import mango


class MangoEdgeDetector(mango):
    def _extract_features(self):
        """Extract edge and histogram features from the mango image."""
        img = self._load_image("train", self.image)
        img_resized = cv2.resize(np.array(img), self.SIZE)

        self.edges = self._get_edges(img_resized).flatten()
        self.histogram = self._get_histogram(img_resized).flatten()

    def _get_edges(self, img):
        """Compute and normalize edges using the Sobel operator."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobel_x, sobel_y)
        return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _get_histogram(self, img):
        """Calculate the color histogram for each channel of the image."""
        return np.concatenate(
            [cv2.calcHist([img], [i], None, [256], [0, 256]) for i in range(3)]
        )

    def __iter__(self):
        """Yield features for iteration over the detector's results."""
        yield from super().__iter__()
        yield from self._map_features("edge", self.edges)
        yield from self._map_features("hist", self.histogram)
