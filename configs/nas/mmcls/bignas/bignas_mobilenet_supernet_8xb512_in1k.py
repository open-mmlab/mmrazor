_base_ = [
    'mmrazor::_base_/settings/imagenet_bs1024_spos.py',
    'mmcls::_base_/default_runtime.py',
]

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

supernet = dict(
    type='mmrazor.SearchableImageClassifier',
    # data_preprocessor=_base_.preprocess_cfg,
    backbone=dict(
        type='BigNASMobileNet', last_out_channels_range=[1280, 1408, 8]),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=1408,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)))

# !autoslim algorithm config
num_samples = 2
model = dict(
    type='mmrazor.BigNAS',
    num_samples=num_samples,
    architecture=supernet,
    data_preprocessor=data_preprocessor,
    distiller=dict(
        type='mmrazor.ConfigurableDistiller',
        teacher_recorders=dict(
            fc=dict(type='mmrazor.ModuleOutputs', source='head.fc')),
        student_recorders=dict(
            fc=dict(type='mmrazor.ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='mmrazor.KLDivergence', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(recorder='fc', from_student=True),
                preds_T=dict(recorder='fc', from_student=False)))),
    mutators=dict(
        channel_mutator=dict(type='mmrazor.BigNASChannelMutator'),
        value_mutator=dict(type='mmrazor.DynamicValueMutator')),
    resizer_cfg=dict(
        input_resizer=dict(type='mmrazor.DynamicInputResizer'),
        mutable_shape=dict(
            type='mmrazor.OneShotMutableValue',
            value_list=[(192, 192), (224, 224), (288, 288), (320, 320)],
            default_value=(224, 224))))

model_wrapper_cfg = dict(
    type='mmrazor.BigNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)
