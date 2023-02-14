# yapf: disable
# flake8: noqa
#############################################################################
# You have to fill these options.

_base_ = 'mmcls::mobilenet_v2/mobilenet-v2_8xb32_in1k.py'  # config to pretrain your model
pretrained_path = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'  # path of pretrained model

interval = 25  # interval between pruning two channels.
prune_mode = 'act'  # prune mode, one of ['act' 'flops']
lr_ratio = 0.1125  # ratio to decrease lr rate to make training stable

target_flop_ratio = 0.65  # the flop rato of target pruned model.
input_shape = [1, 3, 224, 224]  # input shape
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
            default_args=dict(detla_type=prune_mode, ),
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
