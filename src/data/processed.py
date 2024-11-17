import os
import joblib
from . import csv_data, features


class MangoProcessor:
    CSV_DIR = "../data/processed"
    SCALER_PATH = "../data/scaler.pkl"

    def __init__(self, dataset):
        os.makedirs(self.CSV_DIR, exist_ok=True)
        self.raw_data = {
            "train": dataset.train_data,
            "validation": dataset.validation_data,
        }

        self.scaler = self._load_scaler()
        self.train_data = self._create_csv("train")
        self.validation_data = self._create_csv("validation")

    def apply_pca(self, n_components=0.95):
        """Compute the principal components for the image features."""
        self.train_pca, self.pca_n = features.apply_pca(self.train_data, n_components)
        self.validation_pca, _ = features.apply_pca(self.validation_data, self.pca_n)
        print(f"PCA components: {self.train_pca.shape[1]}")

    def _process(self, split):
        """Process the image set to extract feature data."""
        raw_data = self.raw_data[split]
        feature_data = [dict(features(image, split=split)) for image in raw_data]

        if split == "train":
            scaled, self.scaler = features.get_scaled_features(feature_data)
            self._save_scaler(self.scaler)
        else:
            scaled, _ = features.get_scaled_features(feature_data, scaler=self.scaler)

        return scaled

    def process_image(self, image_path):
        """Process the image to extract feature data."""
        feature_data = [dict(features(image_path))]
        scaled, _ = features.get_scaled_features(feature_data, scaler=self.scaler)
        return scaled

    def _create_csv(self, split):
        """Create a CSV file from the processed data."""
        return csv_data(
            os.path.join(self.CSV_DIR, f"{split}.csv"),
            data_generator=lambda: self._process(split),
        )

    def _save_scaler(self, scaler):
        """Save the scaler to a file."""
        joblib.dump(scaler, self.SCALER_PATH)

    def _load_scaler(self):
        """Load the scaler from a file if it exists."""
        if os.path.exists(self.SCALER_PATH):
            return joblib.load(self.SCALER_PATH)
        return None
