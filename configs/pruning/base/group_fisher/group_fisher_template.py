import os

# yapf: disable
# flake8: noqa
#############################################################################
# You have to fill these args.

_base_ = os.environ['BaseConfig']  # config to pretrain your model
pretrained_path = os.environ['PratrainedPath']  # path of pretrained model

interval = int(os.environ.get('Interval', 10)) # interval between pruning two channels.
prune_mode = os.environ.get('PruneMode', 'act') # prune mode, one of ['act' 'flops']
lr_ratio = float(os.environ.get('LrRatio', 0.1))  # ratio to decrease lr rate to make training stable

target_flop_ratio=float(os.environ.get('TargetFlopRatio',0.5)) # the flop rato of target pruned model.
input_shape = os.environ.get('InputShape', (1, 3, 224, 224))  # input shape
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
