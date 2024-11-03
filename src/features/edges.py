import cv2
import numpy as np
from src import Mango


class EdgesMango(Mango):
    def _extract_features(self):
        """Load, resize the image, and extract edges and histogram features."""
        img = self._load_original("train", self.image)
        img_resized = cv2.resize(np.array(img), self.SIZE)

        # Extract features
        self.edges = self._detect_edges(img_resized).flatten()
        self.histogram = self._calculate_histogram(img_resized).flatten()

        # Flatten and concatenate features
        return np.concatenate((self.edges, self.histogram))

    def _detect_edges(self, img):
        """Detect edges in the image using the Sobel operator."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobel_x, sobel_y)
        return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _calculate_histogram(self, img):
        """Calculate and return the histogram for each color channel."""
        return np.concatenate(
            [cv2.calcHist([img], [i], None, [256], [0, 256]) for i in range(3)]
        )

    def __iter__(self):
        """Yield label and features for iteration."""
        yield from super().__iter__()
        for i in range(len(self.edges)):
            yield (f"edge_{i}", int(self.edges[i]))
        for i in range(len(self.histogram)):
            yield (f"hist_{i}", int(self.histogram[i]))
