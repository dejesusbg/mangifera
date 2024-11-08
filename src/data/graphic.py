import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from random import sample
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

    @staticmethod
    def show_samples(train_set, features_set, num_samples=5):
        """Display a set of sample images and their corresponding histograms."""
        num_samples = min(num_samples, len(train_set))

        indices = sample(range(len(train_set)), num_samples)
        selected_images = [(train_set[i], features_set[i]) for i in indices]

        _, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

        for ax, (train_img, features_img) in zip(axes[:, 0], selected_images):
            ax.set_title(train_img["label"])
            ax.imshow(MangoPlotter.load_image("train", train_img))
            ax.axis("off")

        for ax, (train_img, features_img) in zip(axes[:, 1], selected_images):
            ax.set_title("Histogram")
            bins, histogram = MangoPlotter.load_histogram(features_img)
            ax.plot(bins, histogram)
            ax.axis("tight")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_channels(stats_df):
        """Display the mean and standard deviation of the RGB channels."""
        _, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        channels = ["r", "g", "b"]
        for i, ch in enumerate(channels):
            axes[i].hist(stats_df[f"mean_{ch}"], bins=20, color=ch, alpha=0.7)
            axes[i].set_title(f"Mean {ch.upper()} Channel")

            axes[i + 3].hist(stats_df[f"std_dev_{ch}"], bins=20, color=ch, alpha=0.7)
            axes[i + 3].set_title(f"Std Dev {ch.upper()} Channel")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_correlation(stats_df):
        """Display the correlation matrix of the RGB mean and standard deviation."""
        corr_matrix = stats_df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix of RGB Mean and Standard Deviation")
        plt.show()

    @staticmethod
    def show_pca(stats_df):
        """Display the PCA of the RGB mean and standard deviation."""
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(stats_df)

        plt.figure(figsize=(10, 6))
        plt.scatter(pca_components[:, 0], pca_components[:, 1], alpha=0.5, color="blue")
        plt.title("PCA of Image Features")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
