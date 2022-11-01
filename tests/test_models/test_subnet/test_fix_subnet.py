# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch.nn as nn

from mmrazor.models import *  # noqa:F403,F401
from mmrazor.models.mutables import OneShotMutableOP, OneShotMutableValue
from mmrazor.registry import MODELS
from mmrazor.structures import export_fix_subnet, load_fix_subnet
from mmrazor.utils import FixMutable

MODELS.register_module()


class MockModel(nn.Module):

    def __init__(self):
        super().__init__()
        convs1 = nn.ModuleDict({
            'conv1': nn.Conv2d(3, 8, 1),
            'conv2': nn.Conv2d(3, 8, 1),
            'conv3': nn.Conv2d(3, 8, 1),
        })
        convs2 = nn.ModuleDict({
            'conv1': nn.Conv2d(8, 16, 1),
            'conv2': nn.Conv2d(8, 16, 1),
            'conv3': nn.Conv2d(8, 16, 1),
        })

        self.mutable1 = OneShotMutableOP(convs1)
        self.mutable2 = OneShotMutableOP(convs2)

    def forward(self, x):
        x = self.mutable1(x)
        x = self.mutable2(x)
        return x


class MockModelWithDerivedMutable(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.source_mutable = OneShotMutableValue([2, 3, 4], default_value=3)
        self.derived_mutable = self.source_mutable * 2


class TestFixSubnet(TestCase):

    def test_load_fix_subnet(self):
        # fix subnet is str
        fix_subnet = 'tests/data/test_models/test_subnet/mockmodel_subnet.yaml'  # noqa: E501
        model = MockModel()

        load_fix_subnet(model, fix_subnet)

        # fix subnet is dict
        fix_subnet = {
            'mutable1': {
                'chosen': 'conv1'
            },
            'mutable2': {
                'chosen': 'conv2'
            },
        }

        model = MockModel()
        load_fix_subnet(model, fix_subnet)

        model = MockModel()
        load_fix_subnet(model, fix_subnet)

        with pytest.raises(TypeError):
            # type int is not supported.
            model = MockModel()
            load_fix_subnet(model, fix_subnet=10)

        model = MockModel()
        fix_subnet.pop('mutable1')
        with pytest.raises(RuntimeError):
            load_fix_subnet(model, fix_subnet)

    def test_export_fix_subnet(self):
        # get FixSubnet
        fix_subnet = {
            'mutable1': {
                'chosen': 'conv1'
            },
            'mutable2': {
                'chosen': 'conv2'
            },
        }

        model = MockModel()
        load_fix_subnet(model, fix_subnet)

        with pytest.raises(AssertionError):
            exported_fix_subnet: FixMutable = export_fix_subnet(model)

        model = MockModel()
        model.mutable1.current_choice = 'conv1'
        model.mutable2.current_choice = 'conv2'
        exported_fix_subnet = export_fix_subnet(model)

        mutable1_dump_chosen = exported_fix_subnet['mutable1']
        mutable2_dump_chosen = exported_fix_subnet['mutable2']

        mutable1_chosen_dict = dict(chosen=mutable1_dump_chosen.chosen)
        mutable2_chosen_dict = dict(chosen=mutable2_dump_chosen.chosen)

        exported_fix_subnet['mutable1'] = mutable1_chosen_dict
        exported_fix_subnet['mutable2'] = mutable2_chosen_dict
        self.assertDictEqual(fix_subnet, exported_fix_subnet)

    def test_export_fix_subnet_with_derived_mutable(self) -> None:
        model = MockModelWithDerivedMutable()
        fix_subnet = export_fix_subnet(model)
        self.assertDictEqual(
            fix_subnet, {'source_mutable': model.source_mutable.dump_chosen()})

        fix_subnet['source_mutable'] = dict(
            fix_subnet['source_mutable']._asdict())
        fix_subnet['source_mutable']['chosen'] = 4
        load_fix_subnet(model, fix_subnet)
        assert model.source_mutable.current_choice == 4
        assert model.derived_mutable.current_choice == 8

        model = MockModelWithDerivedMutable()
        fix_subnet = export_fix_subnet(model, dump_derived_mutable=True)
        self.assertDictEqual(
            fix_subnet, {
                'source_mutable': model.source_mutable.dump_chosen(),
                'derived_mutable': model.derived_mutable.dump_chosen()
            })

        fix_subnet['source_mutable'] = dict(
            fix_subnet['source_mutable']._asdict())
        fix_subnet['source_mutable']['chosen'] = 2
        load_fix_subnet(model, fix_subnet)
        assert model.source_mutable.current_choice == 2
        assert model.derived_mutable.current_choice == 4
