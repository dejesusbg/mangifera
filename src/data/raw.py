import os
import kagglehub
from src import CSVData


class MangoDataset:
    DATASET_NAME = "adrinbd/unripe-ripe-rotten-mango"
    LABELS = ["Ripe", "Rotten"]
    CSV_DIR = "../data/raw"

    def __init__(self):
        # Ensure the dataset is downloaded and retrieve the path
        self.path = self._initialize_dataset()
        # Create datasets for training and validation
        self.train_data = self._create_csv_dataset("train")
        self.validation_data = self._create_csv_dataset("validation")

    def _initialize_dataset(self):
        """Check if the dataset directory exists and download the dataset if not."""
        if not os.path.exists(self.CSV_DIR):
            os.makedirs(self.CSV_DIR)
            dataset_path = self._download_dataset()
            CSVData.save_dataset_path(dataset_path)
            return dataset_path
        return CSVData.get_saved_dataset_path()

    @classmethod
    def _download_dataset(cls):
        """Download the dataset from Kaggle."""
        return kagglehub.dataset_download(cls.DATASET_NAME)

    def _create_csv_dataset(self, split):
        """Create a CSV dataset for the specified split (train/validation)."""
        return CSVData(
            os.path.join(self.CSV_DIR, f"{split}.csv"),
            data_generator=lambda: self._load_data(split),
        )

    def _load_data(self, split):
        """Load data from the dataset for the specified split."""
        data = []
        # Iterate through the labels to gather filenames
        for label in self.LABELS:
            label_folder = os.path.join(self.path, "Dataset", split, label)
            if os.path.exists(label_folder):
                # Extend the data list with filenames and their labels
                data.extend(
                    {"filename": filename, "label": label}
                    for filename in os.listdir(label_folder)
                )
        return data
