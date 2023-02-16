_base_ = 'mmcls::resnet/resnet50_8xb32_in1k.py'
custom_imports = dict(imports=['projects'])
architecture = _base_.model
pretrained_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa
architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture.update({
    'data_preprocessor': _base_.data_preprocessor,
})
data_preprocessor = None

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    interval=25,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            default_args=dict(normalization_type='flop', ),
        ),
    ),
)
model_wrapper_cfg = dict(
    type='mmrazor.GroupFisherDDP',
    broadcast_buffers=False,
)
# update optimizer

optim_wrapper = dict(optimizer=dict(lr=0.004, ))
param_scheduler = None

custom_hooks = [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=25,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=[1, 3, 224, 224],
        ),
        save_ckpt_delta_thr=[0.75, 0.50],
    ),
]

# original
"""
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
        _scope_='mmcls'))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[30, 60, 90],
    gamma=0.1,
    _scope_='mmcls')
"""

import os

# yapf: disable
# flake8: noqa
#############################################################################
# You have to fill these args.

_base_ = 'mmcls::resnet/resnet50_8xb32_in1k.py'  # config to pretrain your model
pretrained_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth' # path of pretrained model

interval = 25 # interval between pruning two channels.
prune_mode = 'act' # prune mode, one of ['act' 'flops']
lr_ratio = 0.04  # ratio to decrease lr rate to make training stable

target_flop_ratio = 0.5 # the flop rato of target pruned model.
input_shape = [1,3,224,224]  # input shape
##############################################################################
# yapf: enable

architecture = _base_.model

if hasattr(_base_, 'data_preprocessor'):
    architecture.update({'data_preprocessor': _base_.data_preprocessor})
    data_preprocessor = None

architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture['_scope_'] = _base_.default_scope

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    interval=interval,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            default_args=dict(normalization_type=prune_mode, ),
        ),
    ),
)

model_wrapper_cfg = dict(
    type='mmrazor.GroupFisherDDP',
    broadcast_buffers=False,
)

_original_lr_ = _base_.optim_wrapper.optimizer.lr
optim_wrapper = dict(optimizer=dict(lr=_original_lr_ * lr_ratio))

custom_hooks = getattr(_base_, 'custom_hooks', []) + [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=interval,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=input_shape,
        ),
        save_ckpt_thr=[target_flop_ratio],
    ),
]
