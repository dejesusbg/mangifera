import os
from .. import csv_data, features


class MangoProcessor:
    CSV_DIR = "../data/processed"

    def __init__(self, image_set=[]):
        os.makedirs(self.CSV_DIR, exist_ok=True)
        self.image_set = image_set
        self.features = self._create_csv("features", self._process)
        self.pca = self._create_csv("pca", self._get_pca)

    def _process(self):
        """Process images to extract features."""
        return [dict(features(image)) for image in self.image_set]

    def _get_pca(self):
        """Get the principal components of the image features."""
        return features.get_pca_features(self.features)

    def _create_csv(self, split, data_generator):
        """Create a CSV file using the processed data."""
        return csv_data(
            os.path.join(self.CSV_DIR, f"{split}.csv"),
            data_generator=data_generator,
        )
