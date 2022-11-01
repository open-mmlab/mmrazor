_base_ = [
    'mmrazor::_base_/settings/imagenet_bs2048_autoslim_pil.py',
    'mmcls::_base_/models/mobilenet_v2_1x.py',
    'mmcls::_base_/default_runtime.py',
]

supernet = _base_.model
supernet.backbone.widen_factor = 1.5
supernet.head.in_channels = 1920

# !dataset config
# ==========================================================================
# data preprocessor
data_preprocessor = dict(
    type='ImgDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True)

# !autoslim algorithm config
# ==========================================================================

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SlimmableNetwork',
    architecture=supernet,
    data_preprocessor=data_preprocessor,
    mutator=dict(
        type='SlimmableChannelMutator',
        channel_unit_cfg=dict(
            type='SlimmableChannelUnit',
            units='tests/data/MBV2_slimmable_config.json'),
        parse_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss'))))

model_wrapper_cfg = dict(
    type='mmrazor.SlimmableNetworkDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

optim_wrapper = dict(accumulative_counts=3)

val_cfg = dict(type='mmrazor.SlimmableValLoop')
