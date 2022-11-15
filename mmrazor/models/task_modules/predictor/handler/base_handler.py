# Copyright (c) OpenMMLab. All rights reserved.
from joblib import dump, load


class BaseHandler:
    """Base class for a handler.

    Note:
        The handler works through a specific machine leanring algorithm,
        and is designed for predicting the evaluation metric of a model.
    """

    def __init__(self) -> None:
        pass

    def fit(self, train_data, train_label):
        """Training the model of handler."""
        pass

    def predict(self, test_data):
        """Predicting the metric using the model of handler."""
        pass

    def load(self, path):
        """Load pretrained weights for the handler."""
        self.model = load(path)

    def save(self, path):
        """Save the handler and return saved path for diff suffix."""
        path += f'_{self.__class__.__name__}.joblib'.lower()
        dump(self.model, path)
        return path
