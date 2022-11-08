# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import numpy as np
import scipy.stats as stats

from mmrazor.registry import TASK_UTILS
from mmrazor.structures import export_fix_subnet
from .handler import RBFHandler


@TASK_UTILS.register_module()
class MetricPredictor:
    """A predictor for predicting evaluation metrics in different tasks.

    Args:
        handler_cfg (dict): Config to build a predict handler.
        search_groups (dict) : The search_groups of the specified supernet.
        train_samples (int): Num of training samples for the handler.
            Defaults to 2.
        handler_ckpt (str, optional): Path to handler's checkpoint. If given,
            predictor will load weights directly instead of handler training.
        encoding_type (str, optional): Type of how to encode the search space
            to integer bit-string. Defaults to `onehot`.
        score_key (str): Specify one metric in evaluation results to score
            models. Defaults to 'accuracy_top-1'.
    """

    def __init__(self,
                 handler_cfg: Dict,
                 search_groups: Dict,
                 train_samples: int = 2,
                 handler_ckpt: str = None,
                 encoding_type: str = 'onehot',
                 score_key: str = 'accuracy_top-1',
                 **kwargs):
        self.handler_cfg = handler_cfg
        self.handler = TASK_UTILS.build(handler_cfg)

        assert encoding_type in [
            'normal', 'onehot'
        ], ('encoding_type must be `normal` or `onehot`.'
            f'Got `{encoding_type}`.')
        if isinstance(self.handler, RBFHandler):
            encoding_type = 'normal'
        self.encoding_type = encoding_type

        self.search_groups = search_groups
        self.train_samples = train_samples
        self.handler_ckpt = handler_ckpt

        self.score_key_list = [score_key] + ['anticipate']
        self.initialize = False

    def predict(self, model, predict_args: Dict = dict()) -> Dict[str, float]:
        """Predict the evaluation metric of input model using the handler.

        Args:
            model: input model.
            predict_args (Dict, optional): predict args for predictor.

        Returns:
            Dict[str, float]: evaluation metric of the model.
        """
        metric: Dict[str, float] = {}
        if not self.initialize or predict_args.get('anticipate', False):
            raise AssertionError(
                'Call evaluator to get metric instead of predictor.')

        if self.initialize:
            model = export_fix_subnet(model)
            data = self.preprocess(np.array([self.model2vector(model)]))
            score = float(np.squeeze(self.handler.predict(data)))
            if metric.get(self.score_key_list[0], None):
                metric.update({self.score_key_list[1]: score})
            else:
                metric.update({self.score_key_list[0]: score})
        return metric

    def model2vector(self, model: dict) -> Dict[str, list]:
        """Convert the input model to N-dims vector.

        Args:
            model (Dict[str, list]): input model.

        Returns:
            Dict[str, list]: converted vector.
        """
        index = 0
        vector_dict: Dict[str, list] = \
            dict(normal_vector=[], onehot_vector=[])

        for choice in model.values():
            assert len(self.search_groups[index]) == 1
            choices = self.search_groups[index][0].choices
            onehot = np.zeros(len(choices), dtype=np.int)
            _chosen_index = choices.index(choice)
            onehot[_chosen_index] = 1

            vector_dict['normal_vector'].extend([_chosen_index])
            vector_dict['onehot_vector'].extend(onehot)
            index += 1

        return vector_dict

    def vector2model(self, vector: np.array) -> Dict[str, str]:
        """Convert the N-dims vector to original model.

        Args:
            vector (numpy.array): input vector which represents the model.

        Returns:
            Dict[str, str]: converted model.
        """
        start = 0
        model = {}
        for key, value in self.search_groups.items():
            if self.encoding_type == 'onehot':
                index = np.where(vector[start:start +
                                        len(value[0].choices)] == 1)[0][0]
                start += len(value)
            else:
                index = vector[start]
                start += 1
            chosen = value[0].choices[int(index)]
            model[key] = chosen

        return model

    @staticmethod
    def get_correlation(prediction: np.array,
                        label: np.array) -> List[np.array]:
        """Compute the correlations between prediction and ground-truth label.

        Args:
            prediction (numpy.array): predict vector.
            label (numpy.array): ground-truth label.

        Returns:
            List[numpy.array]: coefficients of correlations between predicton
                and ground-truth label.
        """
        rmse = np.sqrt(((prediction - label)**2).mean())
        rho, _ = stats.spearmanr(prediction, label)
        tau, _ = stats.kendalltau(prediction, label)
        return [rmse, rho, tau]

    def preprocess(self, data: List[Dict[str, list]]) -> np.array:
        """Preprocess the data, convert it into np.array format.

        Args:
            data (List[Dict[str, list]]): input data for training.

        Returns:
            numpy.array: input data in numpy.array format.
        """
        if self.encoding_type == 'normal':
            data = np.array([x['normal_vector'] for x in data])
        else:
            data = np.array([x['onehot_vector'] for x in data])
        return data

    def fit(self, data: List[Dict[str, list]], label: np.array) -> None:
        """Training the handler using the structure information of a model. The
        weights of handler will be fixed after that.

        Args:
            data (List[Dict[str, list]]): input data for training.
            label (numpy.array): input label for training.
        """
        data = self.preprocess(data)
        self.handler.fit(data, label)
        self.initialize = True

    def load_checkpoint(self) -> None:
        """Load checkpoint for handler."""
        self.handler.load(self.handler_ckpt)
        self.initialize = True

    def save_checkpoint(self, path: str) -> str:
        """Save checkpoint of handler and return saved path for diff suffix.

        Args:
            path (str): save path for the handler.

        Returns:
            (str): specific checkpoint path of the current handler.
        """
        return self.handler.save(path)
