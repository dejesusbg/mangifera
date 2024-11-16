import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from random import sample
from . import csv_data


class MangoPlotter:
    SIZE = (224, 224)

    @staticmethod
    def load_image(split, image):
        """Load and resize the original image from the the specified split."""
        path = csv_data.get_dataset_path()
        path = os.path.join(path, "Dataset", split, image["label"], image["filename"])
        img_resized = Image.open(path).resize(MangoPlotter.SIZE)
        return np.array(img_resized)

    @staticmethod
    def load_histogram(image):
        """Load the histogram data from the image."""
        bins = np.arange(256)
        histogram = [[image[f"hist_{i + j*256}"] for i in range(256)] for j in range(3)]
        return bins, histogram

    @staticmethod
    def show_samples(images, features, num_samples=5):
        """Display a set of sample images and their corresponding histograms."""
        num_samples = min(num_samples, len(images))

        indices = sample(range(len(images)), num_samples)
        selected_images = [(images.iloc[i], features.iloc[i]) for i in indices]

        _, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

        for ax, (image, features) in zip(axes[:, 0], selected_images):
            ax.set_title(image["label"])
            ax.imshow(MangoPlotter.load_image("train", image))
            ax.axis("off")

        for ax, (image, features) in zip(axes[:, 1], selected_images):
            ax.set_title("Histogram")
            bins, histograms = MangoPlotter.load_histogram(features)
            for hist, color in zip(histograms, ["r", "g", "b"]):
                ax.plot(bins, hist, color=color, alpha=0.7)
            ax.axis("tight")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_channels(stats):
        """Display the mean and standard deviation of the RGB channels."""
        _, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        channels = ["r", "g", "b"]
        for i, ch in enumerate(channels):
            axes[i].hist(stats[f"mean_{ch}"], bins=20, color=ch, alpha=0.7)
            axes[i].set_title(f"Mean {ch.upper()} Channel")

            axes[i + 3].hist(stats[f"std_dev_{ch}"], bins=20, color=ch, alpha=0.7)
            axes[i + 3].set_title(f"Std Dev {ch.upper()} Channel")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_correlation(stats):
        """Display the correlation matrix of the RGB mean and standard deviation."""
        corr_matrix = stats.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix of RGB Mean and Standard Deviation")
        plt.show()

    @staticmethod
    def show_pca(x, y):
        """Display a scatter plot of the PCA components of the image features."""
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.5, color="blue")
        plt.title("PCA of Image Features")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
