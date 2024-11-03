from src import Graphic


class MangoFeatures:
    SIZE = (64, 64)

    def __init__(self, image):
        self.image = image
        self.label = int(image["label"] == "Ripe")
        self.filename = image["filename"]
        self.features = self._extract_features()

    @staticmethod
    def _load_original(split, image):
        """Load the original image from the dataset based on its path."""

        return Graphic.load_original(split, image)

    def __iter__(self):
        """Yield filename and label for iteration."""
        yield ("filename", self.filename)
        yield ("label", self.label)

    def __str__(self):
        """Return a string representation of the image."""
        return str(self.image)
