import os
from .. import csv_data, features


class MangoProcessor:
    CSV_DIR = "../data/processed"

    def __init__(self, image_set=[]):
        os.makedirs(self.CSV_DIR, exist_ok=True)
        self.image_set = image_set
        self.data = self._create_csv()

    def _process(self):
        """Process images to extract features."""
        return [dict(features(image)) for image in self.image_set]

    def _create_csv(self):
        """Create a CSV file using the processed data."""
        return csv_data(
            os.path.join(self.CSV_DIR, f"features.csv"),
            data_generator=lambda: self._process(),
        )
