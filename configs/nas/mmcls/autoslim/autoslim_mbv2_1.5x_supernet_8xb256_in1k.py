_base_ = [
    '../../../_base_/settings/imagenet_bs2048_autoslim_pil.py',
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
    bgr_to_rgb=True,
)

# !autoslim algorithm config
num_random_samples = 2
model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='AutoSlim',
    num_random_samples=num_random_samples,
    architecture=supernet,
    data_preprocessor=data_preprocessor,
    distiller=dict(
        type='ConfigurableDistiller',
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(recorder='fc', from_student=True),
                preds_T=dict(recorder='fc', from_student=False)))),
    mutator=dict(
        type='OneShotChannelMutator',
        channel_unit_cfg=dict(
            type='OneShotMutableChannelUnit',
            default_args=dict(
                candidate_choices=list(i / 12 for i in range(2, 13)),
                choice_mode='ratio',
                divisor=8)),
        parse_cfg=dict(
            type='ChannelAnalyzer',
            demo_input=(1, 3, 224, 224),
            tracer_type='BackwardTracer')))

model_wrapper_cfg = dict(
    type='mmrazor.AutoSlimDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)

# learning policy
max_epochs = 50
param_scheduler = dict(end=max_epochs)

# train, val, test setting
train_cfg = dict(max_epochs=max_epochs)
val_cfg = dict(type='mmrazor.SubnetValLoop')
