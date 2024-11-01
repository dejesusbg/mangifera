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
        """Process and return a list of images as dictionaries."""
        return [dict(Mango(image)) for image in self.image_set]

    def get_processed_data(self):
        """Return processed data."""
        return self.processed_data

    def load_original(self, split, image):
        """Load the original image."""
        return Mango.load_original(split, image)

    def _extract_values(self, image):
        """Extract edges and histogram values from the image."""
        values = list(image.values())
        size = int((len(values) - (255 * 3) - 1) ** 0.5)

        edges_bound = size * size + 1
        edges, histogram = values[1:edges_bound], values[edges_bound:]
        return edges, histogram, size

    def load_processed(self, image):
        """Load and reshape the processed image into a 2D array."""
        img_values, _, size = self._extract_values(image)
        return np.array(img_values).reshape((size, size))

    def load_histogram(self, image):
        """Load and plot the color histogram of the image."""
        _, img_values, _ = self._extract_values(image)
        bins = np.arange(256)

        # Create sublists for each color channel
        colors = ["blue", "green", "red"]
        histograms = [img_values[i : i + 256] for i in range(0, 768, 256)]

        # Plot histograms
        plt.figure(figsize=(10, 5))
        for color, hist in zip(colors, histograms):
            plt.plot(bins, hist, color=color, label=f"{color.capitalize()} Channel")

        plt.title("Color Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        # plt.show()

        return plt
