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
        self.preprocess()

    def detect_edges(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        edges = cv2.magnitude(sobel_x, sobel_y)
        return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def calculate_histogram(self, img):
        hist = []
        for i in range(3):
            channel_hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist.append(channel_hist)

        return np.concatenate(hist)

    def preprocess(self):
        img = self.load_original("train", self.image)
        img = np.array(img)
        img = cv2.resize(img, self.SIZE)

        self.edges = self.detect_edges(img)
        edges_flat = self.edges.flatten()
        self.histogram = self.calculate_histogram(img)
        histogram_flat = self.histogram.flatten()

        self.features = np.concatenate((edges_flat, histogram_flat))

    def __iter__(self):
        yield ("label", self.label)
        for index, feature in enumerate(self.features):
            yield (f"{index}", int(feature))

    def __str__(self):
        return str(self.image)

    @staticmethod
    def load_original(use, image):
        path = CSVData.get_saved_dataset_path()
        label = image["label"]
        filename = image["filename"]
        image_path = os.path.join(path, "Dataset", use, label, filename)

        return Image.open(image_path)
