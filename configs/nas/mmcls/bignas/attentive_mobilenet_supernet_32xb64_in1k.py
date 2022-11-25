_base_ = [
    'mmcls::_base_/default_runtime.py',
    'mmrazor::_base_/settings/imagenet_bs2048_bignas.py',
    'mmrazor::_base_/nas_backbones/bignas_mobilenetv3_supernet.py',
]

supernet = dict(
    _scope_='mmrazor',
    type='SearchableImageClassifier',
    backbone=dict(
        type='AttentiveMobileNetV3',
        arch_setting=_base_.arch_setting,
        norm_cfg=dict(type='DynamicBatchNorm2d', momentum=0.0),
        act_cfg=dict(type='Swish')),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=1984,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
    input_resizer_cfg=dict(
        input_resizer=dict(type='DynamicInputResizer'),
        mutable_shape=dict(
            type='OneShotMutableValue',
            value_list=[[192, 192], [224, 224], [256, 256], [288, 288]],
            default_value=[224, 224])),
    connect_head=dict(connect_with_backbone='backbone.last_mutable'),
)

model = dict(
    _scope_='mmrazor',
    type='BigNAS',
    num_samples=_base_.num_samples,
    drop_path_rate=0.2,
    backbone_dropout_stages=[6, 7],
    architecture=supernet,
    data_preprocessor=_base_.data_preprocessor,
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
    mutators=dict(
        channel_mutator=dict(
            type='mmrazor.OneShotChannelMutator',
            channel_unit_cfg={
                'type': 'OneShotMutableChannelUnit',
                'default_args': {
                    'unit_predefined': True
                }
            },
            parse_cfg={'type': 'Predefined'}),
        value_mutator=dict(type='DynamicValueMutator')))

model_wrapper_cfg = dict(
    type='mmrazor.BigNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)
