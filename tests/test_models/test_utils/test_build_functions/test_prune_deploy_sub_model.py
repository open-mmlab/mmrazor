# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

from mmrazor.models.utils import PruneDeploySubModel
from ....data.models import MMClsResNet18
from .test_prune_sub_model import PruneAlgorithm


class TestPruneDeploySubModel(TestCase):

    def test_deploy_wrapper(self):
        model = MMClsResNet18()

        # get structure
        algorithm = PruneAlgorithm(copy.deepcopy(model))
        algorithm.random_prune()
        strucutrue = algorithm.mutator.current_choices

        wrapper = PruneDeploySubModel(
            copy.deepcopy(model), strucutrue, divisor=1)
        print(wrapper)

        wrapper = PruneDeploySubModel(
            copy.deepcopy(model), strucutrue, divisor=8)
        print(wrapper)
