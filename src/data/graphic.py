import os
import numpy as np
from PIL import Image
from .. import csv_data


class MangoPlotter:
    SIZE = (224, 224)

    @staticmethod
    def load_image(split, img, grayscale=False):
        """Load and resize the original image from the the specified split."""
        path = csv_data.get_dataset_path()
        path = os.path.join(path, "Dataset", split, img["label"], img["filename"])
        img_resized = Image.open(path).resize(MangoPlotter.SIZE)

        if grayscale:
            img_resized = img_resized.convert("L")

        return img_resized

    @staticmethod
    def load_histogram(img):
        """Load histogram data from the image."""
        bins = np.arange(256)
        histogram = [img[f"hist_{i}"] for i in range(256)]
        return bins, histogram
