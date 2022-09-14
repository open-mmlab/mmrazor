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
        type='ZSKTGenerator', img_size=32, latent_dim=256,
        hidden_channels=128),
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1.1.relu'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.1.relu'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.1.relu'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.1.relu'),
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            res34_bb_s1=dict(
                type='ModuleOutputs', source='res34.backbone.layer1.2.relu'),
            res34_bb_s2=dict(
                type='ModuleOutputs', source='res34.backbone.layer2.3.relu'),
            res34_bb_s3=dict(
                type='ModuleOutputs', source='res34.backbone.layer3.5.relu'),
            res34_bb_s4=dict(
                type='ModuleOutputs', source='res34.backbone.layer4.2.relu'),
            res34_fc=dict(type='ModuleOutputs', source='res34.head.fc')),
        distill_losses=dict(
            loss_s1=dict(type='ATLoss', loss_weight=250.0),
            loss_s2=dict(type='ATLoss', loss_weight=250.0),
            loss_s3=dict(type='ATLoss', loss_weight=250.0),
            loss_s4=dict(type='ATLoss', loss_weight=250.0),
            loss_kl=dict(
                type='KLDivergence', loss_weight=2.0, reduction='mean')),
        loss_forward_mappings=dict(
            loss_s1=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s1', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='res34_bb_s1', record_idx=1)),
            loss_s2=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s2', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='res34_bb_s2', record_idx=1)),
            loss_s3=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s3', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='res34_bb_s3', record_idx=1)),
            loss_s4=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s4', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='res34_bb_s4', record_idx=1)),
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='res34_fc')))),
    generator_distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            res34_fc=dict(type='ModuleOutputs', source='res34.head.fc')),
        distill_losses=dict(
            loss_kl=dict(
                type='KLDivergence',
                loss_weight=-2.0,
                reduction='mean',
                teacher_detach=False)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='res34_fc')))),
    student_iter=10)

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
        optimizer=dict(type='SGD', lr=0.1, weight_decay=0.0005, momentum=0.9)),
    generator=dict(optimizer=dict(type='Adam', lr=1e-3)))
auto_scale_lr = dict(base_batch_size=16)

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
    batch_size=16, sampler=dict(type='InfiniteSampler', shuffle=True))
val_dataloader = dict(batch_size=16)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=100, max_keep_ckpts=2))

log_processor = dict(by_epoch=False)
# Must set diff_rank_seed to True!
randomness = dict(seed=None, diff_rank_seed=True)
