_base_ = 'mmcls::mobilenet_v2/mobilenet-v2_8xb32_in1k.py'
custom_imports = dict(imports=['projects'])
architecture = _base_.model
pretrained_path = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'  # noqa
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
            default_args=dict(detla_type='act', ),
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
        save_ckpt_delta_thr=[0.65, 0.33],
    ),
]

# original
"""
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.045,
        momentum=0.9,
        weight_decay=4e-05,
        _scope_='mmcls'))
param_scheduler = dict(
    type='StepLR', by_epoch=True, step_size=1, gamma=0.98, _scope_='mmcls')
"""
