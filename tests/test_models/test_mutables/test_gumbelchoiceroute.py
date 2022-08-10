# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
import torch.nn as nn

from mmrazor.models import *  # noqa:F403,F401
from mmrazor.registry import MODELS

MODELS.register_module(name='torchConv2d', module=nn.Conv2d, force=True)


class TestGumbelChoiceRoute(TestCase):

    def test_forward_arch_param(self):
        edges_dict = nn.ModuleDict({
            'first_edge': nn.Conv2d(32, 32, 3, 1, 1),
            'second_edge': nn.Conv2d(32, 32, 5, 1, 2),
            'third_edge': nn.Conv2d(32, 32, 7, 1, 3),
            'fourth_edge': nn.MaxPool2d(3, 1, 1),
            'fifth_edge': nn.AvgPool2d(3, 1, 1),
        })

        gumbel_choice_route_cfg = dict(
            type='GumbelChoiceRoute',
            edges=edges_dict,
            tau=1.0,
            hard=True,
            with_arch_param=True,
        )

        # test with_arch_param = True
        GumbelChoiceRoute = MODELS.build(gumbel_choice_route_cfg)

        arch_param = nn.Parameter(torch.randn(len(edges_dict)))
        assert len(arch_param) == 5
        GumbelChoiceRoute.set_temperature(1.0)

        x = [torch.randn(4, 32, 64, 64) for _ in range(5)]

        output = GumbelChoiceRoute.forward_arch_param(
            x=x, arch_param=arch_param)
        assert output is not None

        # test with_arch_param = False
        new_gumbel_choice_route_cfg = gumbel_choice_route_cfg.copy()
        new_gumbel_choice_route_cfg['with_arch_param'] = False

        new_gumbel_choice_route = MODELS.build(new_gumbel_choice_route_cfg)

        arch_param = nn.Parameter(torch.randn(len(edges_dict)))
        output = new_gumbel_choice_route.forward_arch_param(
            x=x, arch_param=arch_param)
        assert output is not None

        new_gumbel_choice_route.fix_chosen(chosen=['first_edge'])

    def test_forward_fixed(self):
        edges_dict = nn.ModuleDict({
            'first_edge': nn.Conv2d(32, 32, 3, 1, 1),
            'second_edge': nn.Conv2d(32, 32, 5, 1, 2),
            'third_edge': nn.Conv2d(32, 32, 7, 1, 3),
            'fourth_edge': nn.MaxPool2d(3, 1, 1),
            'fifth_edge': nn.AvgPool2d(3, 1, 1),
        })

        gumbel_choice_route_cfg = dict(
            type='GumbelChoiceRoute',
            edges=edges_dict,
            tau=1.0,
            hard=True,
            with_arch_param=True,
        )

        # test with_arch_param = True
        GumbelChoiceRoute = MODELS.build(gumbel_choice_route_cfg)

        GumbelChoiceRoute.fix_chosen(
            chosen=['first_edge', 'second_edge', 'fifth_edge'])
        assert GumbelChoiceRoute.is_fixed is True

        x = [torch.randn(4, 32, 64, 64) for _ in range(3)]
        output = GumbelChoiceRoute.forward_fixed(x)
        assert output is not None
        assert GumbelChoiceRoute.num_choices == 3

        # after is_fixed = True, call fix_chosen
        with pytest.raises(AttributeError):
            GumbelChoiceRoute.fix_chosen(chosen=['first_edge'])
