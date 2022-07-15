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
channel_cfg_paths = [
    'https://download.openmmlab.com/mmrazor/v1/autoslim/autoslim_mbv2_subnet_8xb256_in1k_flops-220M_acc-71.4_20220715-9c288f3b_subnet_cfg.yaml',  # noqa: E501
    'https://download.openmmlab.com/mmrazor/v1/autoslim/autoslim_mbv2_subnet_8xb256_in1k_flops-320M_acc-72.73_20220715-9aa8f8ae_subnet_cfg.yaml',  # noqa: E501
    'https://download.openmmlab.com/mmrazor/v1/autoslim/autoslim_mbv2_subnet_8xb256_in1k_flops-530M_acc-74.23_20220715-aa8754fe_subnet_cfg.yaml'  # noqa: E501
]

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SlimmableNetwork',
    architecture=supernet,
    data_preprocessor=data_preprocessor,
    channel_cfg_paths=channel_cfg_paths,
    mutator=dict(
        type='SlimmableChannelMutator',
        mutable_cfg=dict(type='SlimmableMutableChannel'),
        tracer_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss'))))

model_wrapper_cfg = dict(
    type='mmrazor.SlimmableNetworkDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

optim_wrapper = dict(accumulative_counts=len(channel_cfg_paths))

val_cfg = dict(type='mmrazor.SlimmableValLoop')
