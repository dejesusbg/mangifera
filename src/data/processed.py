import os
from . import csv_data, features


class MangoProcessor:
    CSV_DIR = "../data/processed"

    def __init__(self, image_set=[]):
        os.makedirs(self.CSV_DIR, exist_ok=True)
        self.image_set = image_set
        self.features_data = self._create_csv("features", self._process)

    def get_pca(self, split="default", hist_components=50):
        """Get the principal components of the image features."""
        filename = f"pca_{split}"
        return self._create_csv(filename, lambda: self._process_pca(hist_components))

    def _process(self):
        """Process images to extract features."""
        return [dict(features(image)) for image in self.image_set]

    def _process_pca(self, hist_components):
        """Get the principal components of the image features."""
        return features.get_pca_features(self.features_data, hist_components)

    @classmethod
    def _create_csv(cls, split, data_generator):
        """Create a CSV file using the processed data."""
        return csv_data(
            os.path.join(cls.CSV_DIR, f"{split}.csv"),
            data_generator=data_generator,
        )
