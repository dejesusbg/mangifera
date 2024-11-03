import os
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from PIL import Image
from src import CSVData


class MangoPlot:
    @staticmethod
    def load_original(split, image):
        """Load the original image from the dataset based on its path."""
        path = CSVData.get_saved_dataset_path()
        label = image["label"]
        filename = image["filename"]
        image_path = os.path.join(path, "Dataset", split, label, filename)
        return Image.open(image_path)

    @staticmethod
    def _get_size():
        """Get the size of the image."""
        from src import Mango

        return Mango.SIZE

    @classmethod
    def _extract_values(cls, image, is_compressed=False):
        """Extract edges and histogram values from the image."""
        size = cls._get_size()
        edges = []
        histogram = []

        if is_compressed:
            histogram = [image[f"hist_{i}"] for i in range(256)]
        else:
            edges = [image[f"edge_{i}"] for i in range(size[0] * size[1])]
            histogram = [image[f"hist_{i}"] for i in range(256 * 3)]

        return edges, histogram

    @classmethod
    def load_processed(cls, image):
        """Load and reshape the processed image into a 2D array."""
        edges, _ = cls._extract_values(image)
        return np.array(edges).reshape(cls._get_size())

    @classmethod
    def load_histogram(cls, image, is_compressed=False):
        """Load the color histogram of the image."""
        _, histograms = cls._extract_values(image, is_compressed)
        bins = np.arange(256)

        if is_compressed:
            plt.plot(bins, histograms, color="orange", label="Gray Channel")
        else:
            # Create sublists for each color channel
            colors = ["blue", "green", "red"]
            histograms = [histograms[i : i + 256] for i in range(0, 768, 256)]

            # Plot histograms
            plt.figure(figsize=(10, 5))
            for color, hist in zip(colors, histograms):
                label = f"{color.capitalize()} Channel"
                plt.plot(bins, hist, color=color, label=label, alpha=0.7)

        plt.title(f"Color Histogram of {image['filename']}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()

        return plt

    @classmethod
    def show_samples(cls, proc_set, orig_set, num_samples=5):
        num_samples = min(num_samples, len(proc_set))

        indices = sample(range(len(proc_set)), num_samples)
        selected_images = [(orig_set[i], proc_set[i]) for i in indices]

        _, axes = plt.subplots(2, num_samples, figsize=(15, 6))

        for ax, (orig_img, proc_img) in zip(axes[0], selected_images):
            ax.imshow(cls.load_original("train", orig_img))
            ax.axis("off")

        for ax, (orig_img, proc_img) in zip(axes[1], selected_images):
            ax.set_title(proc_img["label"])
            ax.imshow(cls.load_processed(proc_img))
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    @classmethod
    def show_histogram_samples(cls, edges_set, stats_set, num_samples=1):
        num_samples = min(num_samples, len(edges_set))

        indices = sample(range(len(edges_set)), num_samples)

        for i in indices:
            edges = cls.load_histogram(edges_set[i])
            stats = cls.load_histogram(stats_set[i], True)

            edges.show()
            stats.show()
