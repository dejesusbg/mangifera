import os
import kagglehub
from . import csv_data, features


class MangoDataset:
    DATASET_NAME = "adrinbd/unripe-ripe-rotten-mango"
    LABELS = ("Ripe", "Rotten")
    CSV_DIR = "../data/raw"

    def __init__(self):
        self.path = self._find_dataset()
        self.train_data = self._create_csv("train")
        self.validation_data = self._create_csv("validation")

    def get_labels(self):
        """Generate the labels for the image features"""
        return (
            features.get_encoded_labels(self.train_data),
            features.get_encoded_labels(self.validation_data),
        )

    def _find_dataset(self):
        """Check for dataset existence and download it if not present."""
        if not os.path.exists(self.CSV_DIR):
            os.makedirs(self.CSV_DIR)
            dataset_path = self._download()
            csv_data.save_dataset_path(dataset_path)
            return dataset_path
        return csv_data.get_dataset_path()

    def _load_dataset(self, split):
        """Load dataset files and their labels for the specified split."""
        data = []
        for label in self.LABELS:
            label_folder = os.path.join(self.path, "Dataset", split, label)
            if os.path.exists(label_folder):
                for filename in os.listdir(label_folder):
                    data.append({"label": label, "filename": filename})
        return data

    def _create_csv(self, split):
        """Create a CSV file from the dataset for the specified split."""
        return csv_data(
            os.path.join(self.CSV_DIR, f"{split}.csv"),
            data_generator=lambda: self._load_dataset(split),
        )

    @classmethod
    def _download(cls):
        """Download the dataset from Kaggle."""
        return kagglehub.dataset_download(cls.DATASET_NAME)
