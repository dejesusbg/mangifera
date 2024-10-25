import os
from PIL import Image
import kagglehub


class MangoDataset:
    def __init__(self):
        self.dataset_name = "adrinbd/unripe-ripe-rotten-mango"
        self.path = self.download()

    def download(self):
        path = kagglehub.dataset_download(self.dataset_name)
        print("Path to dataset files:", path)
        return path

    def load(self, use, classification):

        folder = os.path.join(self.path, "Dataset", use, classification)

        images = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path):
                img = Image.open(img_path)
                images.append(img)

        return images
