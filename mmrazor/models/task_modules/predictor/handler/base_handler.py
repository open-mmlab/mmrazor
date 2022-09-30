# Copyright (c) OpenMMLab. All rights reserved.
from joblib import dump, load


class BaseHandler():
    """Base handler."""

    def __init__(self) -> None:
        pass

    def fit(self, train_data, train_label):
        pass

    def predict(self, test_data):
        pass

    def load(self, path):
        """Load pretrained weights for the handler."""
        self.model = load(path)

    def save(self, path):
        """Save the handler and return saved path for diff suffix."""
        path += f'_{self.__class__.__name__}.joblib'.lower()
        dump(self.model, path)
        return path
