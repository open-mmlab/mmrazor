_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]
optim_wrapper = dict(
    _delete_=True,
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(
        type='OptimWrapper',
        optimizer=dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=3e-4),
        clip_grad=dict(max_norm=5, norm_type=2)),
    mutator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=3e-4, weight_decay=1e-3)))

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

train_cfg = dict(
    by_epoch=True,
    max_epochs=120,
    val_interval=1)

data_preprocessor = {'type': 'mmcls.ClsDataPreprocessor'}

custom_hooks = [dict(type='DMCPSubnetHook')]

# model settings
model = dict(
    _scope_='mmrazor',
    type='DMCP',
    architecture=dict(
        cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=False),
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
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss'))),
    arch_start_train=10000,
    step_freq=500,
    distillation_times=20000,
    target_flops=2000)

model_wrapper_cfg = dict(
    type='mmrazor.DMCPDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)