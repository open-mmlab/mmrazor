_base_ = [
    'mmcls::_base_/default_runtime.py',
    'mmrazor::_base_/settings/imagenet_bs2048_ofa.py',
    'mmrazor::_base_/nas_backbones/ofa_mobilenetv3_supernet.py',
]

supernet = dict(
    _scope_='mmrazor',
    type='SearchableImageClassifier',
    data_preprocessor=_base_.data_preprocessor,
    backbone=_base_.nas_backbone,
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
    input_resizer_cfg=_base_.input_resizer_cfg,
    connect_head=dict(connect_with_backbone='backbone.last_mutable_channels'),
)

model = dict(
    _scope_='mmrazor',
    type='BigNAS',
    drop_path_rate=0.2,
    backbone_dropout_stages=[6, 7],
    architecture=supernet,
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
    mutators=dict(type='mmrazor.NasMutator'))

model_wrapper_cfg = dict(
    type='mmrazor.BigNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)
