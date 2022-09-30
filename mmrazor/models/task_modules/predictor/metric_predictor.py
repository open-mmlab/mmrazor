# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import numpy as np
import scipy.stats as stats

from mmrazor.registry import TASK_UTILS
from mmrazor.structures import export_fix_subnet
from .base_predictor import BasePredictor
from .handler import MLPHandler, RBFHandler


@TASK_UTILS.register_module()
class MetricPredictor(BasePredictor):
    """Metric predictor.

    Args:
        handler_type (str): The chosen type for different handler.
        search_groups(dict) : search_groups of the specified super net.
        pool_size(int) : Total nums of training samples if necessary.
        handler_cfg (dict, optional): Optional parameter for predictor.
        fit_cfg(dict, optional): Training parameters. Only supported for
            MLP predictor.
        pretrained(str, optional): Path to predictor's weights. If given,
            predictor will load the specified weights and skip
            training phase.
        evaluation(str, optional): If not None, compute the correlations
            between prediction and true label, used for evaluate the
            predictor's performance. 'simple' only evaluate the final
            samples, 'complex' whill evaluate samples in the
            candidate_pool. Defaults to None.
        retrain_samples(dict, optional): Given the training samples for
            predictor.
        evaluate_samples(dict, optional): Given the predicting samples for
            predictor.
        encoding_type(str, optional): how to encode the search space to
            integer bit-string. Default is onehot.
    """

    def __init__(self,
                 handler_cfg: dict,
                 search_groups,
                 score_key,
                 train_samples: int = 2,
                 fit_cfg: dict = None,
                 pretrained=None,
                 evaluation=None,
                 evaluate_samples=None,
                 encoding_type='onehot',
                 **kwargs):
        super().__init__(handler_cfg=handler_cfg)

        if isinstance(self.handler, RBFHandler):
            encoding_type = 'normal'
        assert encoding_type in [
            'normal', 'onehot'
        ], ('encoding_type must be `normal` or `onehot`.'
            f'Got `{encoding_type}`.')
        self.encoding_type = encoding_type

        self.search_groups = search_groups
        self.train_samples = train_samples
        self.pretrained = pretrained
        self.evaluation = evaluation
        assert evaluation in [None, 'simple', 'complex'
                              ], (f'Not support evaluation mode {evaluation}.')

        self.fit_cfg = fit_cfg
        self.evaluate_samples = evaluate_samples
        self.score_key_list = [score_key] + ['anticipate']
        self.initialize = False

    def predict(self, model, predict_args=dict()):
        """Predict the candidate's performance."""
        metric = {}
        if not self.initialize or predict_args.get('anticipate', False):
            raise AssertionError(
                'Call evaluator to get metric instead of predictor.')

        if self.initialize:
            candidate = export_fix_subnet(model)
            input = self.preprocess(np.array([self.spec2feats(candidate)]))
            score = float(np.squeeze(self.handler.predict(input)))
            if metric.get(self.score_key_list[0], None):
                metric.update({self.score_key_list[1]: score})
            else:
                metric.update({self.score_key_list[0]: score})
        return metric

    def spec2feats(self, candidate: dict) -> dict:
        """Convert the candidate to N dimensions vector.

        N is different for different supernet.
        """
        index = 0
        feats_dict: Dict[str, list] = dict(feats=[], onehot_feats=[])

        for choice in candidate.values():
            assert len(self.search_groups[index]) == 1
            _candidates = self.search_groups[index][0].choices
            onehot = np.zeros(len(_candidates))
            _chosen_index = _candidates.index(choice)
            onehot[_chosen_index] = 1

            feats_dict['feats'].extend([_chosen_index])
            feats_dict['onehot_feats'].extend(onehot)
            index += 1

        return feats_dict

    def feats2spec(self, feats, type='onehot'):
        """Convert the N dimensions vector to original candidates.

        feats is the output comes form self.spec2feats.
        """
        fix_subnet = {}
        start = 0
        for key, value in self.search_groups.items():
            if type == 'onehot':
                index = np.where(feats[start:start +
                                       len(value[0].choices)] == 1)[0][0]
                start += len(value)
            else:
                index = feats[start]
                start += 1
            chosen = value[0].choices[int(index)]
            fix_subnet[key] = chosen
        return fix_subnet

    def load_checkpoint(self):
        self.handler.load(self.pretrained)
        self.initialize = True

    def save(self, path):
        """Save predictor and return saved path for diff suffix;"""
        return self.handler.save(path)

    def get_correlation(self, prediction, target):
        """Compute the correlations between prediction and true label."""
        rmse = np.sqrt(((prediction - target)**2).mean())
        rho, _ = stats.spearmanr(prediction, target)
        tau, _ = stats.kendalltau(prediction, target)

        return rmse, rho, tau

    def preprocess(self, input):
        if isinstance(input[0], dict):
            if self.encoding_type == 'normal':
                input = np.array([x['feats'] for x in input])
            else:
                input = np.array([x['onehot_feats'] for x in input])
        return input

    def fit(self, input, target):
        """Training the accuracy predictor."""
        input = self.preprocess(input)
        if isinstance(self.handler, MLPHandler):
            self.handler.check_dimentions(input.shape[1])
            self.handler.fit(x=input, y=target, **self.fit_cfg)
        else:
            self.handler.fit(input, target)

        self.initialize = True
