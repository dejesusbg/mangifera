import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from random import sample
from . import csv_data


class MangoPlotter:
    SIZE = (224, 224)

    @staticmethod
    def load_image(path):
        """Load and resize the original image."""
        img_resized = Image.open(path).resize(MangoPlotter.SIZE)
        return np.array(img_resized)

    @staticmethod
    def load_dataset_image(image, split):
        """Load and resize the original image from the the specified split."""
        path = csv_data.get_dataset_path()
        path = os.path.join(path, "Dataset", split, image["label"], image["filename"])
        return MangoPlotter.load_image(path)

    @staticmethod
    def load_histogram(image):
        """Load the histogram data from the image."""
        bins = np.arange(256)
        histogram = [[image[f"hist_{i + j*256}"] for i in range(256)] for j in range(3)]
        return bins, histogram

    @staticmethod
    def show_dataset(train_set, validation_set, num_samples=5):
        """Display a set of sample images."""
        train_samples = sample(train_set, num_samples)
        validation_samples = sample(validation_set, num_samples)

        _, axes = plt.subplots(2, num_samples, figsize=(12, 6))
        axes = axes.flatten()

        for i, image in enumerate(train_samples):
            axes[i].imshow(MangoPlotter.load_dataset_image(image, "train"))
            axes[i].set_title(f"Entrenamiento: {image['label']}")
            axes[i].axis("off")

        for i, image in enumerate(validation_samples):
            axes[i + num_samples].imshow(
                MangoPlotter.load_dataset_image(image, "validation")
            )
            axes[i + num_samples].set_title(f"Validación: {image['label']}")
            axes[i + num_samples].axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_data_distribution(train_set, validation_set):
        """Display the distribution of the data in the training and validation sets."""
        train_labels = [image["label"] for image in train_set]
        validation_labels = [image["label"] for image in validation_set]

        plt.figure(figsize=(12, 2))
        sns.countplot(train_labels, label="Entrenamiento")
        sns.countplot(validation_labels, label="Validación")
        plt.title("Distribución de Datos")
        plt.xlabel("#")
        plt.ylabel("Categoría")
        plt.legend()
        plt.show()

    @staticmethod
    def show_samples(images, features, num_samples=5):
        """Display a set of sample images and their corresponding histograms."""
        num_samples = min(num_samples, len(images))

        indices = sample(range(len(images)), num_samples)
        selected_images = [(images.iloc[i], features.iloc[i]) for i in indices]

        _, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

        for ax, (image, features) in zip(axes[:, 0], selected_images):
            ax.set_title(image["label"])
            ax.imshow(MangoPlotter.load_dataset_image(image, "train"))
            ax.axis("off")

        for ax, (image, features) in zip(axes[:, 1], selected_images):
            ax.set_title("Histograma")
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
            axes[i].set_title(f"Mean {ch.upper()} channel")

            axes[i + 3].hist(stats[f"std_dev_{ch}"], bins=20, color=ch, alpha=0.7)
            axes[i + 3].set_title(f"Std Dev {ch.upper()} channel")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_correlation(stats):
        """Display the correlation matrix of the RGB mean and standard deviation."""
        corr_matrix = stats.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Matriz de Correlación")
        plt.show()

    @staticmethod
    def show_pca(features, labels, x=0, y=1):
        """Display the principal components of the features."""
        labels = pd.DataFrame(labels)
        df = pd.DataFrame(data=features.loc[:, [x, y]], index=features.index)
        df = pd.concat((df, labels), axis=1, join="inner")
        v1 = f"Component {x}"
        v2 = f"Component {y}"
        df.columns = [v1, v2, "Label"]
        sns.lmplot(x=v1, y=v2, hue="Label", data=df, fit_reg=False)
        ax = plt.gca()
        ax.set_title("Separación por Componentes Principales")

    @classmethod
    def _plot_scattered_cm(cls, cm_values, labels, axis):
        """Plot confusion matrix values as scatter points on a given axis."""
        for cm, label in zip(cm_values, labels):
            false_rate = cm[1] / sum(cm)
            true_rate = cm[0] / sum(cm)
            axis.scatter(false_rate, true_rate, label=label)

        axis.set_xlabel("False Rate")
        axis.set_ylabel("True Rate")
        axis.set_xlim(-0.01, 0.26)
        axis.set_ylim(0.74, 1.01)
        axis.legend()
        axis.grid(True)

    @classmethod
    def show_scattered_cm(cls, results_df, label):
        """Display confusion matrices grouped by model type and class."""
        filtered_results = results_df[results_df["runtype"] == label]
        grouped_by_model = filtered_results.groupby("model")

        cm_class_0_values = []
        cm_class_1_values = []
        model_labels = []

        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        for model_name, group_data in grouped_by_model:
            cm_class_0 = group_data["cm"].apply(lambda cm: cm[0]).iloc[0]
            cm_class_1 = group_data["cm"].apply(lambda cm: cm[1]).iloc[0][::-1]

            cm_class_0_values.append(cm_class_0)
            cm_class_1_values.append(cm_class_1)
            model_labels.append(model_name)

        cls._plot_scattered_cm(cm_class_0_values, model_labels, axes[0])
        axes[0].set_title("Class 0: Ripe (Positive)")

        cls._plot_scattered_cm(cm_class_1_values, model_labels, axes[1])
        axes[1].set_title("Class 1: Rotten (Negative)")

        plt.suptitle(label)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_results_heatmap(result, label):
        heatmap_data = result.pivot(index="runtype", columns="model", values=label)

        plt.figure(figsize=(12, 2))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="coolwarm_r", cbar=True)
        plt.title(f"Mapa de calor de {label}")
        plt.show()

    @staticmethod
    def show_custom_predict(mango, preprocessor, tests, models):
        """Display the predictions of a set of models on a set of images."""
        predictions_grid = []

        for image in tests:
            row_predictions = []
            for model in models:
                prediction = mango.custom_predict(model, image, preprocessor)

                if isinstance(prediction, np.ndarray):
                    prediction = prediction.item()

                row_predictions.append(int(prediction > 0.5))
            predictions_grid.append(row_predictions)

        predictions_grid = np.array(predictions_grid)
        cmap = sns.diverging_palette(133, 10, as_cmap=True, s=75, l=75)

        plt.figure(figsize=(12, 4))
        ax = sns.heatmap(
            predictions_grid,
            cbar_kws={"ticks": [0, 1]},
            annot=True,
            cmap=cmap,
            xticklabels=models,
            yticklabels=tests,
            linewidths=2,
        )

        ax.set_title("Predicciones por Modelo e Imagen")
        plt.tight_layout()
        plt.show()
