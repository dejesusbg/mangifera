import os
import cv2
import numpy as np
from PIL import Image
from src.data.csv import CSVData


class MangoImage:
    SIZE = (64, 64)

    def __init__(self, image):
        self.image = image
        self.label = int(image["label"] == "Ripe")
        self.features = self._preprocess()

    def _preprocess(self):
        """Load, resize the image, and extract edges and histogram features."""
        img = self.load_original("train", self.image)
        img_resized = cv2.resize(np.array(img), self.SIZE)

        # Extract features
        edges = self._detect_edges(img_resized)
        histogram = self._calculate_histogram(img_resized)

        # Flatten and concatenate features
        return np.concatenate((edges.flatten(), histogram.flatten()))

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
        yield ("label", self.label)
        for index, feature in enumerate(self.features):
            yield (f"{index}", int(feature))

    def __str__(self):
        """Return a string representation of the image."""
        return str(self.image)

    @staticmethod
    def load_original(split, image):
        """Load the original image from the dataset based on its path."""
        path = CSVData.get_saved_dataset_path()
        label = image["label"]
        filename = image["filename"]
        image_path = os.path.join(path, "Dataset", split, label, filename)
        return Image.open(image_path)
