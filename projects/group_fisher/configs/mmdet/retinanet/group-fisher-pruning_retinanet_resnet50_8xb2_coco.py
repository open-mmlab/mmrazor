_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'
custom_imports = dict(imports=['projects'])

architecture = _base_.model

architecture.backbone.frozen_stages = -1

if hasattr(_base_, 'data_preprocessor'):
    architecture.update({'data_preprocessor': _base_.data_preprocessor})
    data_preprocessor = None

pretrained_path = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'  # noqa
architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    interval=10,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            default_args=dict(detla_type='flop', ),
        ),
    ),
)

model_wrapper_cfg = dict(
    type='mmrazor.GroupFisherDDP',
    broadcast_buffers=False,
)

optim_wrapper = dict(optimizer=dict(lr=0.002))

custom_hooks = [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=10,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=[1, 3, 1333, 800],
        ),
        save_ckpt_delta_thr=[0.75, 0.5],
    ),
]
