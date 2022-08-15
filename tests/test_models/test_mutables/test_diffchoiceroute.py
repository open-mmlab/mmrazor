# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
import torch.nn as nn

from mmrazor.models import *  # noqa:F403,F401
from mmrazor.registry import MODELS

MODELS.register_module(name='torchConv2d', module=nn.Conv2d, force=True)


class TestDiffChoiceRoute(TestCase):

    def test_forward_arch_param(self):
        edges_dict = nn.ModuleDict()
        edges_dict.add_module('first_edge', nn.Conv2d(32, 32, 3, 1, 1))
        edges_dict.add_module('second_edge', nn.Conv2d(32, 32, 5, 1, 2))
        edges_dict.add_module('third_edge', nn.MaxPool2d(3, 1, 1))
        edges_dict.add_module('fourth_edge', nn.MaxPool2d(5, 1, 2))
        edges_dict.add_module('fifth_edge', nn.MaxPool2d(7, 1, 3))

        diff_choice_route_cfg = dict(
            type='DiffChoiceRoute',
            edges=edges_dict,
            with_arch_param=True,
        )

        # test with_arch_param = True
        diffchoiceroute = MODELS.build(diff_choice_route_cfg)
        arch_param = nn.Parameter(torch.randn(len(edges_dict)))

        x = [torch.randn(4, 32, 64, 64) for _ in range(5)]
        output = diffchoiceroute.forward_arch_param(x=x, arch_param=arch_param)
        assert output is not None

        # test with_arch_param = False
        new_diff_choice_route_cfg = diff_choice_route_cfg.copy()
        new_diff_choice_route_cfg['with_arch_param'] = False

        new_diff_choice_route = MODELS.build(new_diff_choice_route_cfg)
        arch_param = nn.Parameter(torch.randn(len(edges_dict)))
        output = new_diff_choice_route.forward_arch_param(
            x=x, arch_param=arch_param)
        assert output is not None

        new_diff_choice_route.fix_chosen(chosen=['first_edge'])

        # test sample choice
        arch_param = nn.Parameter(torch.randn(len(edges_dict)))
        new_diff_choice_route.sample_choice(arch_param)

        # test dump_chosen
        with pytest.raises(AssertionError):
            new_diff_choice_route.dump_chosen()

    def test_forward_fixed(self):
        edges_dict = nn.ModuleDict({
            'first_edge': nn.Conv2d(32, 32, 3, 1, 1),
            'second_edge': nn.Conv2d(32, 32, 5, 1, 2),
            'third_edge': nn.Conv2d(32, 32, 7, 1, 3),
            'fourth_edge': nn.MaxPool2d(3, 1, 1),
            'fifth_edge': nn.AvgPool2d(3, 1, 1),
        })

        diff_choice_route_cfg = dict(
            type='DiffChoiceRoute',
            edges=edges_dict,
            with_arch_param=True,
        )

        # test with_arch_param = True
        diffchoiceroute = MODELS.build(diff_choice_route_cfg)

        diffchoiceroute.fix_chosen(
            chosen=['first_edge', 'second_edge', 'fifth_edge'])
        assert diffchoiceroute.is_fixed is True

        x = [torch.randn(4, 32, 64, 64) for _ in range(5)]
        output = diffchoiceroute.forward_fixed(x)
        assert output is not None
        assert diffchoiceroute.num_choices == 3

        # after is_fixed = True, call fix_chosen
        with pytest.raises(AttributeError):
            diffchoiceroute.fix_chosen(chosen=['first_edge'])
