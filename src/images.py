import os
import cv2
import numpy as np
from PIL import Image


class MangoManager:
    path = ""

    def __init__(self, path):
        MangoManager.path = path

    def create_image(self, image_data):
        return MangoImage(image_data)


class MangoImage:
    SIZE = (64, 64)

    def __init__(self, image):
        self.image = image
        self.label = image["label"]
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
            hist.append(cv2.calcHist([img], [i], None, [256], [0, 256]))

        return np.concatenate(hist)

    def preprocess(self):
        img = self.load("train", self.image)
        img = np.array(img)
        img = cv2.resize(img, self.SIZE)

        self.edges = self.detect_edges(img)
        edges = self.edges.flatten()

        self.histogram = self.calculate_histogram(img)
        histogram = self.histogram.flatten()

        self.features = np.concatenate((edges, histogram))

    def __iter__(self):
        yield ("label", self.image["label"])
        yield ("filename", self.image["filename"])
        for index, feature in enumerate(self.features):
            yield (f"{index}", feature)

    def __str__(self):
        return str(self.image)

    @staticmethod
    def load(use, image):
        image_path = os.path.join(
            MangoManager.path, "Dataset", use, image["label"], image["filename"]
        )
        img = Image.open(image_path)
        return img
