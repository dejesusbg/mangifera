import os
from src import csv_data


class MangoProcessor:
    CSV_DIR = "../data/processed"

    def __init__(self, image_set=[]):
        os.makedirs(self.CSV_DIR, exist_ok=True)
        self.image_set = image_set
        self.edges_data = self._create_csv("edges", False)
        self.stats_data = self._create_csv("stats", True)

    def _process(self, compressed):
        """Process images to extract features."""
        from src import get_edges, get_stats

        features = get_stats if compressed else get_edges
        return [dict(features(image)) for image in self.image_set]

    def _create_csv(self, split, compressed):
        """Create a CSV file for the specified split using the processing method."""
        return csv_data(
            os.path.join(self.CSV_DIR, f"{split}.csv"),
            data_generator=lambda: self._process(compressed),
        )
