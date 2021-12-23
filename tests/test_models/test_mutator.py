# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch

from mmrazor.models.builder import ARCHITECTURES, MUTATORS


def test_one_shot_mutator():
    model_cfg = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(type='SearchableShuffleNetV2', widen_factor=1.0),
        neck=dict(type='mmcls.GlobalAveragePooling'),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=1000,
            in_channels=1024,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ),
    )

    architecture_cfg = dict(
        type='MMClsArchitecture',
        model=model_cfg,
    )

    mutator_cfg = dict(
        type='OneShotMutator',
        placeholder_mapping=dict(
            all_blocks=dict(
                type='OneShotOP',
                choices=dict(
                    shuffle_3x3=dict(type='ShuffleBlock', kernel_size=3),
                    shuffle_5x5=dict(type='ShuffleBlock', kernel_size=5),
                    shuffle_7x7=dict(type='ShuffleBlock', kernel_size=7),
                    shuffle_xception=dict(type='ShuffleXception'),
                ))))

    imgs = torch.randn(16, 3, 224, 224)
    label = torch.randint(0, 1000, (16, ))

    architecture_cfg_ = deepcopy(architecture_cfg)
    architecture = ARCHITECTURES.build(architecture_cfg_)

    mutator_cfg_ = deepcopy(mutator_cfg)
    mutator = MUTATORS.build(mutator_cfg_)

    mutator.prepare_from_supernet(architecture)
    assert hasattr(mutator, 'search_spaces')
    assert len(mutator.search_spaces) > 0

    # test forward
    subnet_dict = mutator.sample_subnet()
    mutator.set_subnet(subnet_dict)
    losses = architecture(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0

    # test set_chosen_subnet
    mutator.set_chosen_subnet(subnet_dict)
    search_spaces = mutator.search_spaces
    for key in search_spaces.keys():
        assert 'chosen' in search_spaces[key].keys()

    # test mutation and crossover
    subnet_dict1 = mutator.sample_subnet()
    subnet_dict2 = mutator.sample_subnet()
    mutation_subnet_dict = mutator.mutation(subnet_dict1)
    crossover_subnet_dict = mutator.crossover(subnet_dict1, subnet_dict2)
    assert isinstance(mutation_subnet_dict, dict)
    assert len(mutation_subnet_dict) > 0
    assert isinstance(crossover_subnet_dict, dict)
    assert len(crossover_subnet_dict) > 0


def test_differentiable_mutator():
    architecture_cfg = dict(
        type='MMClsArchitecture',
        model=dict(
            type='mmcls.ImageClassifier',
            backbone=dict(
                type='DartsBackbone',
                in_channels=3,
                base_channels=8,
                num_layers=4,
                num_nodes=4,
                stem_multiplier=3,
                out_indices=(3, )),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=10,
                in_channels=128,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5),
            ),
        ),
    )

    mutator_cfg = dict(
        type='DartsMutator',
        placeholder_mapping=dict(
            node=dict(
                type='DifferentiableOP',
                with_arch_param=True,
                choices=dict(
                    zero=dict(type='DartsZero'),
                    skip_connect=dict(
                        type='DartsSkipConnect',
                        norm_cfg=dict(type='BN', affine=False)),
                    max_pool_3x3=dict(
                        type='DartsPoolBN',
                        pool_type='max',
                        norm_cfg=dict(type='BN', affine=False)),
                    avg_pool_3x3=dict(
                        type='DartsPoolBN',
                        pool_type='avg',
                        norm_cfg=dict(type='BN', affine=False)),
                    sep_conv_3x3=dict(
                        type='DartsSepConv',
                        kernel_size=3,
                        norm_cfg=dict(type='BN', affine=False)),
                    sep_conv_5x5=dict(
                        type='DartsSepConv',
                        kernel_size=5,
                        norm_cfg=dict(type='BN', affine=False)),
                    dil_conv_3x3=dict(
                        type='DartsDilConv',
                        kernel_size=3,
                        norm_cfg=dict(type='BN', affine=False)),
                    dil_conv_5x5=dict(
                        type='DartsDilConv',
                        kernel_size=5,
                        norm_cfg=dict(type='BN', affine=False)),
                )),
            node_edge=dict(
                type='DifferentiableEdge',
                num_chosen=2,
                with_arch_param=False,
            )),
    )

    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    architecture_cfg_ = deepcopy(architecture_cfg)
    architecture = ARCHITECTURES.build(architecture_cfg_)

    mutator_cfg_ = deepcopy(mutator_cfg)
    mutator = MUTATORS.build(mutator_cfg_)

    # test prepare_from_superbet
    mutator.prepare_from_supernet(architecture)
    assert hasattr(mutator, 'search_spaces')
    assert hasattr(mutator, 'arch_params')

    # test forward
    losses = architecture(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0

    # test search_subnet
    # TODO
