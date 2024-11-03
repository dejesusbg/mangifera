import os
from src import CSVData


class MangoProcessor:
    CSV_DIR = "../data/processed"

    def __init__(self, image_set=[]):
        os.makedirs(self.CSV_DIR, exist_ok=True)

        self.image_set = image_set
        self.edges_data = self._create_csv_dataset("edges", False)
        self.stats_data = self._create_csv_dataset("stats", True)

    def _create_csv_dataset(self, split, is_compressed):
        """Create a CSV dataset for the specified type (edges/stats)."""
        return CSVData(
            os.path.join(self.CSV_DIR, f"{split}.csv"),
            data_generator=lambda: self._process(is_compressed),
        )

    def _process(self, is_compressed):
        """Process and return a list of images as dictionaries."""
        from src import EdgesMango, StatsMango

        extract_features = StatsMango if is_compressed else EdgesMango
        return [dict(extract_features(image)) for image in self.image_set]

    def get_processed_data(self):
        """Return processed data."""
        return [self.edges_data, self.stats_data]
