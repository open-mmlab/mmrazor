# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

from mmrazor.models.utils import PruneDeploySubModel
from ....data.models import MMClsResNet18
from .test_prune_sub_model import PruneAlgorithm, get_model_structure


class TestPruneDeploySubModel(TestCase):

    def test_init(self):
        model = MMClsResNet18()

        # get structure
        algorithm = PruneAlgorithm(copy.deepcopy(model))
        algorithm.random_prune()
        strucutrue = algorithm.mutator.current_choices

        wrapper = PruneDeploySubModel(
            copy.deepcopy(model), strucutrue, divisor=1)
        self.assertSequenceEqual(
            list(strucutrue.values()),
            list(get_model_structure(wrapper).values()))

        wrapper = PruneDeploySubModel(
            copy.deepcopy(model), strucutrue, divisor=8)
        self.assertSequenceEqual(
            list(strucutrue.values()),
            list(get_model_structure(wrapper).values()))
