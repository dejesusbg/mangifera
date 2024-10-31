import os
import numpy as np
import matplotlib.pyplot as plt
from src.data.csv import CSVData
from src.data.image import MangoImage as Mango


class MangoProcessor:
    CSV_DIR = "../data/processed"

    def __init__(self, image_set=[]):
        self.image_set = image_set

        os.makedirs(self.CSV_DIR, exist_ok=True)
        self.csv_path = os.path.join(self.CSV_DIR, "data.csv")
        self.processed_data = CSVData(self.csv_path, data_generator=self._process)

    def _process(self):
        return [dict(Mango(image)) for image in self.image_set]

    def get_processed_data(self):
        return self.processed_data

    def load_original(self, use, image):
        return Mango.load_original(use, image)

    def _extract_values(self, image):
        values = list(image.values())
        size = int((len(values) - (255 * 3) - 1) ** 0.5)

        edges_bound = size * size + 1
        edges, histogram = values[1:edges_bound], values[edges_bound:]
        return edges, histogram, size

    def load_processed(self, image):
        img_values, _, size = self._extract_values(image)
        img_array = np.array(img_values).reshape((size, size))
        return img_array

    def load_histogram(self, image):
        _, hist, _ = self._extract_values(image)
        bins = np.arange(256)

        b_hist = hist[0:256]
        g_hist = hist[256:512]
        r_hist = hist[512:768]

        plt.figure(figsize=(10, 5))
        plt.plot(bins, b_hist, color="blue", label="Blue Channel")
        plt.plot(bins, g_hist, color="green", label="Green Channel")
        plt.plot(bins, r_hist, color="red", label="Red Channel")

        plt.title("Color Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
