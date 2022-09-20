# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.mutators import BaseChannelMutator
from ....data.models import LineModel


class TestL1MutableChannelGroup(TestCase):

    def test_init(self):
        model = LineModel()
        mutator = BaseChannelMutator(
            channel_group_cfg={
                'type': 'L1MutableChannelGroup',
                'default_args': {
                    'choice_mode': 'ratio'
                }
            })
        mutator.prepare_from_supernet(model)
        mutator.set_choices(mutator.sample_choices())
        print(mutator.groups)
        print(mutator.mutable_groups)
        print(mutator.choice_template)
