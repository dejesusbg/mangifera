import cv2
import numpy as np
from .. import graphic


class MangoFeatureExtractor:
    SIZE = (64, 64)

    def __init__(self, img):
        self.image = img
        self.label = int(img["label"] == "Ripe")
        self.filename = img["filename"]
        self._extract_features()

    def __repr__(self):
        """Return a string representation of the image."""
        return str(self.image)

    def __iter__(self):
        """Yield filename and label as key-value pairs."""
        yield ("filename", self.filename)
        yield ("label", self.label)

    @classmethod
    def _load_image(cls, split, img):
        """Load and resize the image from the specified split."""
        img = graphic.load_orig_img(split, img)
        img_resized = cv2.resize(np.array(img), cls.SIZE)
        return img_resized

    @staticmethod
    def _map_features(name, features):
        """Yield feature names and their corresponding values."""
        for i in range(len(features)):
            yield (f"{name}_{i}", features[i])
