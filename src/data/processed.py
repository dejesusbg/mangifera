import os
from . import csv_data, features


class MangoProcessor:
    CSV_DIR = "../data/processed"

    def __init__(self, dataset):
        os.makedirs(self.CSV_DIR, exist_ok=True)
        self.raw_data = {
            "train": dataset.train_data,
            "validation": dataset.validation_data,
        }

        self.train_data = self._create_csv("train")
        self.validation_data = self._create_csv("validation")

    def apply_pca(self, n_components=0.95):
        """Compute the principal components for the image features."""
        self.train_pca, n_components = features.apply_pca(self.train_data, n_components)
        self.validation_pca, _ = features.apply_pca(self.validation_data, n_components)
        print(f"PCA components: {self.train_pca.shape[1]}")

    def _process(self, split):
        """Process the image set to extract feature data."""
        raw_data = self.raw_data[split]
        feature_data = [dict(features(split, image)) for image in raw_data]
        return features.get_scaled_features(feature_data)

    def _create_csv(self, split):
        """Create a CSV file from the processed data."""
        return csv_data(
            os.path.join(self.CSV_DIR, f"{split}.csv"),
            data_generator=lambda: self._process(split),
        )
