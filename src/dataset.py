import os
import pandas as pd
import kagglehub


class MangoDataset(tuple):
    DATASET_NAME = "adrinbd/unripe-ripe-rotten-mango"
    LABELS = ["Ripe", "Rotten"]
    CSV_DIR = "dataset/raw"
    PATH_FILE = "dataset/path.txt"

    def __new__(cls):
        path = cls.download_if_needed()
        train = cls.get_data_from_csv(os.path.join(cls.CSV_DIR, "train.csv"))
        validation = cls.get_data_from_csv(os.path.join(cls.CSV_DIR, "validation.csv"))

        return super().__new__(cls, (path, train, validation))

    @classmethod
    def download_if_needed(cls):
        if not os.path.exists(cls.CSV_DIR):
            os.makedirs(cls.CSV_DIR)

            path = cls.download()
            cls.save_dataset_path(path)  # Save the path to a file
            cls.create_csv(path, "train", os.path.join(cls.CSV_DIR, "train.csv"))
            cls.create_csv(
                path, "validation", os.path.join(cls.CSV_DIR, "validation.csv")
            )

            return path
        else:
            return cls.get_existing_dataset_path()

    @classmethod
    def download(cls):
        return kagglehub.dataset_download(cls.DATASET_NAME)

    @classmethod
    def save_dataset_path(cls, path):
        with open(cls.PATH_FILE, "w") as f:
            f.write(path)

    @classmethod
    def get_existing_dataset_path(cls):
        if os.path.exists(cls.PATH_FILE):
            with open(cls.PATH_FILE, "r") as f:
                return f.read().strip()
        return None

    @classmethod
    def create_csv(cls, path, use, csv_file):
        data = cls.get_data(path, use)
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

    @classmethod
    def get_data(cls, path, use):
        data = []

        for label in cls.LABELS:
            folder = os.path.join(path, "Dataset", use, label)
            if os.path.exists(folder):
                filenames = os.listdir(folder)

                for filename in filenames:
                    data.append({"label": label, "filename": filename})

        return data

    @classmethod
    def get_data_from_csv(cls, csv_file):
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            return df.to_dict(orient="records")
        return []
