import os
import kagglehub


class MangoDataset:
    DATASET_NAME = "adrinbd/unripe-ripe-rotten-mango"
    TAGS = ["Ripe", "Rotten"]

    def __init__(self):
        self.path = self.download()
        self.train = self.get_data("train")
        self.validation = self.get_data("validation")

    def download(self):
        return kagglehub.dataset_download(self.DATASET_NAME)

    def get_data(self, use):
        data = []

        for tag in self.TAGS:
            folder = os.path.join(self.path, "Dataset", use, tag)
            filenames = os.listdir(folder)

            for filename in filenames:
                data.append({"tag": tag, "filename": filename})

        return data
