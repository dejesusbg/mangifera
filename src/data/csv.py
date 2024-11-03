import os
import pandas as pd
import time


class CSVData(list):
    PATH_FILE = "../data/path.txt"

    def __init__(self, csv_path, data_generator=lambda: []):
        start_time = time.time()

        self.csv_path = csv_path
        self.data_generator = data_generator
        self.data = self._load_data()
        self._save_data_to_csv()

        print(f"Loaded dataset {csv_path} in {time.time() - start_time:.2f} seconds.")
        super().__init__(self.data)

    def _load_data(self):
        if os.path.exists(self.csv_path):
            return self._read_csv()
        return self.data_generator()

    def _read_csv(self):
        df = pd.read_csv(self.csv_path)
        return df.to_dict(orient="records")

    def _save_data_to_csv(self):
        path = os.path.join(os.getcwd(), self.csv_path)
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)

    @classmethod
    def get_saved_dataset_path(cls):
        if os.path.exists(cls.PATH_FILE):
            with open(cls.PATH_FILE, "r") as f:
                return f.read().strip()
        return None

    @classmethod
    def save_dataset_path(cls, path):
        with open(cls.PATH_FILE, "w") as f:
            f.write(path)
