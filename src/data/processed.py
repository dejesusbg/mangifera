import os
import numpy as np
from PIL import Image
from src.data.csv import CSVData
from src.data.image import MangoImage as Mango


class MangoProcessor:
    CSV_DIR = "../data/processed"

    def __init__(self, path, image_set=None):
        self.image_set = image_set or []
        self.mango = Mango
        Mango.PATH = path

        os.makedirs(self.CSV_DIR, exist_ok=True)
        self.csv_path = os.path.join(self.CSV_DIR, "data.csv")
        self.processed_data = CSVData(self.csv_path, data_generator=self._process)

    def _process(self):
        return [dict(self.mango(image)) for image in self.image_set]

    def get_processed_data(self):
        return self.processed_data

    def load_original(self, use, image):
        return self.mango.load_original(use, image)

    def load_processed(self, image, size=(64, 64)):
        img_values = list(image.values())[1:4097]
        img_array = np.array(img_values).reshape(size)
        return img_array
