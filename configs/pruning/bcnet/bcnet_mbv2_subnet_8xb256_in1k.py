model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1920,
        loss=dict(
            type='LabelSmoothLoss',
            mode='original',
            label_smooth_val=0.1,
            loss_weight=1.0),
        topk=(1, 5),
    ))

algorithm = dict(
    type='BCNet',
    architecture=dict(type='MMClsArchitecture', model=model),
    pruner=dict(
        type='BCNetPruner',
        ratios=(1 / 20, 2 / 20, 3 / 20, 4 / 20, 5 / 20, 6 / 20, 7 / 20, 8 / 20,
                9 / 20, 10 / 20, 11 / 20, 12 / 20, 13 / 20, 14 / 20, 15 / 20,
                16 / 20, 17 / 20, 18 / 20, 19 / 20, 1.0)),
    retraining=True,
    bn_training_mode=False,
    input_shape=None,
    loss_rec_num=100,
    use_complementary=True)

use_ddp_wrapper = True

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy', save_best='auto')

# optimizer
paramwise_cfg = dict(bias_decay_mult=0.0, norm_decay_mult=0.0)
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    nesterov=True,
    weight_decay=5e-5,
    paramwise_cfg=paramwise_cfg)
optimizer_config = None
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-5, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=300)

# checkpoint saving
checkpoint_config = dict(
    interval=1, max_keep_ckpts=10, out_dir='s3://caoweihan/mmrazor_bcnet')
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
