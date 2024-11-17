import os
import pandas as pd
import time


class CSVData(list):
    PATH_FILE = "../data/path.txt"

    def __init__(self, path, data_generator=lambda: []):
        self.path = path
        self.data_generator = data_generator

        start_time = time.time()
        self.data = self._load()
        self._save()

        self._log(start_time)
        super().__init__(self.data)

    def _log(self, start_time):
        """Log the dataset loading time and shape."""
        name = self.path.split("/")[-1].split(".")[0]
        load_time = time.time() - start_time
        num_rows = len(self.data)
        num_cols = len(self.data[0]) if num_rows > 0 else 0

        print(f"Loaded '{name}' in {load_time:.2f}s | {num_rows} rows, {num_cols} cols")

    def _load(self):
        """Load data from the CSV file or generate it if the file does not exist."""
        if os.path.exists(self.path):
            return self._read()
        return self.data_generator()

    def _read(self):
        """Read data from the CSV file and return it as a list of dictionaries."""
        df = pd.read_csv(self.path)
        return df.to_dict(orient="records")

    def _save(self):
        """Save the current data to the CSV file in DataFrame format."""
        df = pd.DataFrame(self.data)
        df.to_csv(self.path, index=False)

    @classmethod
    def get_dataset_path(cls):
        """Retrieve the saved dataset path from the path file, if it exists."""
        if os.path.exists(cls.PATH_FILE):
            with open(cls.PATH_FILE, "r") as f:
                return f.read().strip()
        return None

    @classmethod
    def save_dataset_path(cls, path):
        """Save the dataset path to the designated path file."""
        with open(cls.PATH_FILE, "w") as f:
            f.write(path)
