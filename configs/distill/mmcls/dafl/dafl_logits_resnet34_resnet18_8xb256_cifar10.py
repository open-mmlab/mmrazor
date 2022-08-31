_base_ = [
    'mmcls::_base_/datasets/cifar10_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]

res34_ckpt_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='DAFLDataFreeDistillation',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        # convert image from BGR to RGB
        bgr_to_rgb=False),
    architecture=dict(
        cfg_path='mmcls::resnet/resnet18_8xb16_cifar10.py', pretrained=False),
    teachers=dict(
        res34=dict(
            build_cfg=dict(
                cfg_path='mmcls::resnet/resnet34_8xb16_cifar10.py',
                pretrained=True),
            ckpt_path=res34_ckpt_path)),
    generator=dict(
        type='DAFLGenerator',
        img_size=32,
        latent_dim=1000,
        hidden_channels=128),
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            res34_fc=dict(type='ModuleOutputs', source='res34.head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='KLDivergence', tau=6, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='res34_fc')))),
    generator_distiller=dict(
        type='ConfigurableDistiller',
        teacher_recorders=dict(
            res34_neck_gap=dict(type='ModuleOutputs', source='res34.neck.gap'),
            res34_fc=dict(type='ModuleOutputs', source='res34.head.fc')),
        distill_losses=dict(
            loss_res34_oh=dict(type='OnehotLikeLoss', loss_weight=0.05),
            loss_res34_ie=dict(type='InformationEntropyLoss', loss_weight=5),
            loss_res34_ac=dict(type='ActivationLoss', loss_weight=0.01)),
        loss_forward_mappings=dict(
            loss_res34_oh=dict(
                preds_T=dict(from_student=False, recorder='res34_fc')),
            loss_res34_ie=dict(
                preds_T=dict(from_student=False, recorder='res34_fc')),
            loss_res34_ac=dict(
                feat_T=dict(from_student=False, recorder='res34_neck_gap')))))

# model wrapper
model_wrapper_cfg = dict(
    type='mmengine.MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)

find_unused_parameters = True

# optimizer wrapper
optim_wrapper = dict(
    _delete_=True,
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(optimizer=dict(type='AdamW', lr=1e-1)),
    generator=dict(optimizer=dict(type='AdamW', lr=1e-3)))

auto_scale_lr = dict(base_batch_size=256)

param_scheduler = dict(
    _delete_=True,
    architecture=[
        dict(type='LinearLR', end=500, by_epoch=False, start_factor=0.0001),
        dict(
            type='MultiStepLR',
            begin=500,
            milestones=[100 * 120, 200 * 120],
            by_epoch=False)
    ],
    generator=dict(
        type='LinearLR', end=500, by_epoch=False, start_factor=0.0001))

train_cfg = dict(
    _delete_=True, by_epoch=False, max_iters=250 * 120, val_interval=150)

train_dataloader = dict(
    batch_size=256, sampler=dict(type='InfiniteSampler', shuffle=True))
val_dataloader = dict(batch_size=256)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=75, log_metric_by_epoch=False),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=150, max_keep_ckpts=2))

log_processor = dict(by_epoch=False)
# Must set diff_rank_seed to True!
randomness = dict(seed=None, diff_rank_seed=True)
