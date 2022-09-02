_base_ = [
    'mmcls::_base_/datasets/cifar10_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]

res34_ckpt_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='DataFreeDistillation',
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
        latent_dim=256,
        hidden_channels=128,
        bn_eps=1e-5),
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            res34_fc=dict(type='ModuleOutputs', source='res34.head.fc')),
        distill_losses=dict(loss_kl=dict(type='L1Loss', loss_weight=1.0)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                s_feature=dict(from_student=True, recorder='fc'),
                t_feature=dict(from_student=False, recorder='res34_fc')))),
    generator_distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            res34_fc=dict(type='ModuleOutputs', source='res34.head.fc')),
        distill_losses=dict(loss_l1=dict(type='L1Loss', loss_weight=-1.0)),
        loss_forward_mappings=dict(
            loss_l1=dict(
                s_feature=dict(from_student=True, recorder='fc'),
                t_feature=dict(from_student=False, recorder='res34_fc')))),
    student_iter=5,
    student_train_first=True)

# model wrapper
model_wrapper_cfg = dict(
    type='mmengine.MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True)

# optimizer wrapper
optim_wrapper = dict(
    _delete_=True,
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(
        optimizer=dict(type='SGD', lr=0.1, weight_decay=5e-4, momentum=0.9)),
    generator=dict(optimizer=dict(type='AdamW', lr=1e-3)))

auto_scale_lr = dict(base_batch_size=32)

iter_size = 50
param_scheduler = dict(
    _delete_=True,
    architecture=dict(
        type='MultiStepLR',
        milestones=[100 * iter_size, 200 * iter_size],
        by_epoch=False),
    generator=dict(
        type='MultiStepLR',
        milestones=[100 * iter_size, 200 * iter_size],
        by_epoch=False))

train_cfg = dict(
    _delete_=True, by_epoch=False, max_iters=500 * iter_size, val_interval=250)

train_dataloader = dict(
    batch_size=32, sampler=dict(type='InfiniteSampler', shuffle=True))
val_dataloader = dict(batch_size=32)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=100, max_keep_ckpts=2))

log_processor = dict(by_epoch=False)
# Must set diff_rank_seed to True!
randomness = dict(seed=None, diff_rank_seed=True)
