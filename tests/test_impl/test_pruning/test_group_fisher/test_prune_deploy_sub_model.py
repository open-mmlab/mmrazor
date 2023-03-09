# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from unittest import TestCase

import torch
from mmengine import fileio

from mmrazor import digit_version
from mmrazor.implementations.pruning.group_fisher.prune_deploy_sub_model import \
    GroupFisherDeploySubModel  # noqa
from ....data.models import MMClsResNet18
from .test_prune_sub_model import PruneAlgorithm, get_model_structure


class TestPruneDeploySubModel(TestCase):

    def check_torch_version(self):
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('version of torch < 1.12.0')

    def test_build_sub_model(self):
        self.check_torch_version()
        model = MMClsResNet18()

        parse_cfg = dict(
            _scope_='mmrazor',
            type='ChannelAnalyzer',
            demo_input=(1, 3, 224, 224),
            tracer_type='BackwardTracer')
        # get structure
        algorithm = PruneAlgorithm(copy.deepcopy(model))
        algorithm.random_prune()
        strucutrue = algorithm.mutator.current_choices

        # test divisor
        wrapper = GroupFisherDeploySubModel(
            copy.deepcopy(model), strucutrue, divisor=1, parse_cfg=parse_cfg)
        self.assertSequenceEqual(
            list(strucutrue.values()),
            list(get_model_structure(wrapper).values()))

        wrapper = GroupFisherDeploySubModel(
            copy.deepcopy(model), strucutrue, divisor=8, parse_cfg=parse_cfg)
        self.assertSequenceEqual(
            list(strucutrue.values()),
            list(get_model_structure(wrapper).values()))

        mutable_path = os.path.dirname(__file__) + '/mutable.json'
        fileio.dump(algorithm.mutator.current_choices, mutable_path)
        GroupFisherDeploySubModel(
            copy.deepcopy(model),
            divisor=1,
            mutable_cfg=mutable_path,
            parse_cfg=parse_cfg)
        os.remove(mutable_path)
