# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from unittest import TestCase

from mmengine import fileio

from mmrazor.impl.pruning.group_fisher.prune_deploy_sub_model import \
    GroupFisherDeploySubModel
from ....data.models import MMClsResNet18
from .test_prune_sub_model import PruneAlgorithm, get_model_structure


class TestPruneDeploySubModel(TestCase):

    def test_build_sub_model(self):
        model = MMClsResNet18()

        # get structure
        algorithm = PruneAlgorithm(copy.deepcopy(model))
        algorithm.random_prune()
        strucutrue = algorithm.mutator.current_choices

        # test divisor
        wrapper = GroupFisherDeploySubModel(
            copy.deepcopy(model), strucutrue, divisor=1)
        self.assertSequenceEqual(
            list(strucutrue.values()),
            list(get_model_structure(wrapper).values()))

        wrapper = GroupFisherDeploySubModel(
            copy.deepcopy(model), strucutrue, divisor=8)
        self.assertSequenceEqual(
            list(strucutrue.values()),
            list(get_model_structure(wrapper).values()))

        mutable_path = os.path.dirname(__file__) + '/mutable.json'
        fileio.dump(algorithm.mutator.current_choices, mutable_path)
        GroupFisherDeploySubModel(
            copy.deepcopy(model), divisor=1, mutable_cfg=mutable_path)
        os.remove(mutable_path)
