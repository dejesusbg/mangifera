import os
import numpy as np
from PIL import Image
from src import csv_data


class MangoPlotter:
    @staticmethod
    def _get_img_size():
        """Retrieve the size of the mango images."""
        from src import mango

        return mango.SIZE

    @classmethod
    def _extract_data(cls, img, compressed=False):
        """Extract edges and histogram data from the image."""
        edges, histogram = [], []
        if compressed:
            histogram = [img[f"hist_{i}"] for i in range(256)]
        else:
            size = cls._get_img_size()
            edges = [img[f"edge_{i}"] for i in range(size[0] * size[1])]
            histogram = [img[f"hist_{i}"] for i in range(256 * 3)]
        return edges, histogram

    @classmethod
    def load_orig_img(cls, split, img):
        """Load and resize the original image from the the specified split."""
        path = csv_data.get_dataset_path()
        path = os.path.join(path, "Dataset", split, img["label"], img["filename"])
        img_resized = Image.open(path).resize(cls._get_img_size())
        return img_resized

    @classmethod
    def load_proc_img(cls, img):
        """Load processed image data and reshape it."""
        edges, _ = cls._extract_data(img)
        return np.array(edges).reshape(cls._get_img_size())

    @classmethod
    def load_hist(cls, img, compressed=False):
        """Load histogram data from rgb or grayscale data."""
        _, histogram = cls._extract_data(img, compressed)
        if not compressed:
            histogram = [histogram[i : i + 256] for i in range(0, 768, 256)]
        return histogram

    @classmethod
    def load_all_hists(cls, edges_img, stats_img):
        """Load histograms from both rgb and grayscale data, and prepare for plotting."""
        edges_hist = cls.load_hist(edges_img)
        stats_hist = cls.load_hist(stats_img, True)

        bins = np.arange(256)
        histograms = edges_hist + [stats_hist]
        colors = ["blue", "green", "red", "orange"]
        return bins, colors, histograms

    @staticmethod
    def combine_images(orig_img, proc_img):
        """Combine the original and processed images side by side for comparison."""
        if proc_img.ndim == 2:
            proc_img = np.stack([proc_img] * 3, axis=-1)

        return np.hstack((orig_img, proc_img))
