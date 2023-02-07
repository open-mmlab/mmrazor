_base_ = [
    'mmcls::_base_/default_runtime.py',
    '../../../_base_/settings/imagenet_bs2048_dmcp.py',
]

# model settings
supernet = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            mode='original',
            num_classes=1000,
            label_smooth_val=0.1,
            loss_weight=1.0),
        topk=(1, 5),
    ))

model = dict(
    _scope_='mmrazor',
    type='DMCP',
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
    mutator_cfg=dict(
        type='DMCPChannelMutator',
        channel_unit_cfg=dict(
            type='DMCPChannelUnit', default_args=dict(choice_mode='number')),
        parse_cfg=dict(
            type='ChannelAnalyzer',
            demo_input=(1, 3, 224, 224),
            tracer_type='BackwardTracer')),
    strategy=['max', 'min', 'scheduled_random', 'arch_random'],
    arch_start_train=5000,
    arch_train_freq=500,
    flop_loss_weight=0.1,
    distillation_times=10000,
    target_flops=100)

model_wrapper_cfg = dict(
    type='mmrazor.DMCPDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

randomness = dict(seed=0, diff_rank_seed=True)
