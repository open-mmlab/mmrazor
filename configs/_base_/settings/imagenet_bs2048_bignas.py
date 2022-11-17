# dataset settings
dataset_type = 'mmcls.ImageNet'

# data preprocessor
data_preprocessor = dict(
    type='mmcls.ClsDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
    num_classes=1000,
    batch_augments=dict(
        augments=[
            dict(type='mmcls.Mixup', alpha=0.2),
            dict(type='mmcls.CutMix', alpha=1.0)
        ],
        probs=[0.5, 0.5]))

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

extra_params = dict(
    translate_const=int(224 * 0.45),
    img_mean=tuple(round(x) for x in data_preprocessor['mean']),
)
policies = [
    [
        dict(
            type='mmcls.EqualizeV2',
            prob=0.8,
            magnitude=1,
            extra_params=extra_params),
        dict(
            type='mmcls.ShearY',
            prob=0.8,
            magnitude=4,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.Color',
            prob=0.4,
            magnitude=9,
            extra_params=extra_params),
        dict(
            type='mmcls.EqualizeV2',
            prob=0.6,
            magnitude=3,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.Color',
            prob=0.4,
            magnitude=1,
            extra_params=extra_params),
        dict(
            type='mmcls.RotateV2',
            prob=0.6,
            magnitude=8,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.SolarizeV2',
            prob=0.8,
            magnitude=3,
            extra_params=extra_params),
        dict(
            type='mmcls.EqualizeV2',
            prob=0.4,
            magnitude=7,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.SolarizeV2',
            prob=0.4,
            magnitude=2,
            extra_params=extra_params),
        dict(
            type='mmcls.SolarizeV2',
            prob=0.6,
            magnitude=2,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.Color',
            prob=0.2,
            magnitude=0,
            extra_params=extra_params),
        dict(
            type='mmcls.EqualizeV2',
            prob=0.8,
            magnitude=8,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.EqualizeV2',
            prob=0.4,
            magnitude=8,
            extra_params=extra_params),
        dict(
            type='mmcls.SolarizeAddV2',
            prob=0.8,
            magnitude=3,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.ShearX',
            prob=0.2,
            magnitude=9,
            extra_params=extra_params),
        dict(
            type='mmcls.RotateV2',
            prob=0.6,
            magnitude=8,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.Color',
            prob=0.6,
            magnitude=1,
            extra_params=extra_params),
        dict(
            type='mmcls.EqualizeV2',
            prob=1.0,
            magnitude=2,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.InvertV2',
            prob=0.4,
            magnitude=9,
            extra_params=extra_params),
        dict(
            type='mmcls.RotateV2',
            prob=0.6,
            magnitude=0,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.EqualizeV2',
            prob=1.0,
            magnitude=9,
            extra_params=extra_params),
        dict(type='ShearY', prob=0.6, magnitude=3, extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.Color',
            prob=0.4,
            magnitude=7,
            extra_params=extra_params),
        dict(
            type='mmcls.EqualizeV2',
            prob=0.6,
            magnitude=0,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.PosterizeV2',
            prob=0.4,
            magnitude=6,
            extra_params=extra_params),
        dict(
            type='mmcls.AutoContrastV2',
            prob=0.4,
            magnitude=7,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.SolarizeV2',
            prob=0.6,
            magnitude=8,
            extra_params=extra_params),
        dict(
            type='mmcls.Color',
            prob=0.6,
            magnitude=9,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.SolarizeV2',
            prob=0.2,
            magnitude=4,
            extra_params=extra_params),
        dict(
            type='mmcls.RotateV2',
            prob=0.8,
            magnitude=9,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.RotateV2',
            prob=1.0,
            magnitude=7,
            extra_params=extra_params),
        dict(
            type='mmcls.TranslateYRel',
            prob=0.8,
            magnitude=9,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.ShearX',
            prob=0.0,
            magnitude=0,
            extra_params=extra_params),
        dict(
            type='mmcls.SolarizeV2',
            prob=0.8,
            magnitude=4,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.ShearY',
            prob=0.8,
            magnitude=0,
            extra_params=extra_params),
        dict(
            type='mmcls.Color',
            prob=0.6,
            magnitude=4,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.Color',
            prob=1.0,
            magnitude=0,
            extra_params=extra_params),
        dict(
            type='mmcls.RotateV2',
            prob=0.6,
            magnitude=2,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.EqualizeV2',
            prob=0.8,
            magnitude=4,
            extra_params=extra_params),
        dict(
            type='mmcls.EqualizeV2',
            prob=0.0,
            magnitude=8,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.EqualizeV2',
            prob=1.0,
            magnitude=4,
            extra_params=extra_params),
        dict(
            type='mmcls.AutoContrastV2',
            prob=0.6,
            magnitude=2,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.ShearY',
            prob=0.4,
            magnitude=7,
            extra_params=extra_params),
        dict(
            type='mmcls.SolarizeAddV2',
            prob=0.6,
            magnitude=7,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.PosterizeV2',
            prob=0.8,
            magnitude=2,
            extra_params=extra_params),
        dict(
            type='mmcls.SolarizeV2',
            prob=0.6,
            magnitude=10,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.SolarizeV2',
            prob=0.6,
            magnitude=8,
            extra_params=extra_params),
        dict(
            type='mmcls.EqualizeV2',
            prob=0.6,
            magnitude=1,
            extra_params=extra_params),
    ],
    [
        dict(
            type='mmcls.Color',
            prob=0.8,
            magnitude=6,
            extra_params=extra_params),
        dict(
            type='mmcls.RotateV2',
            prob=0.4,
            magnitude=5,
            extra_params=extra_params),
    ],
]

train_pipeline = [
    dict(type='mmcls.LoadImageFromFile'),
    dict(type='mmcls.RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.AutoAugmentV2', policies=policies),
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='mmcls.LoadImageFromFile'),
    dict(
        type='mmcls.ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bilinear'),
    dict(type='mmcls.CenterCrop', crop_size=224),
    dict(type='mmcls.PackClsInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='mmcls.RepeatAugSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='mmcls.DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
num_samples = 2
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.8, momentum=0.9, weight_decay=0.00001, nesterov=True),
    clip_grad=dict(clip_value=1.0),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
    accumulative_counts=num_samples + 2)

# learning policy
max_epochs = 360
warmup_epochs = 5
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - warmup_epochs,
        eta_min=0,
        by_epoch=True,
        begin=warmup_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='mmrazor.AutoSlimValLoop', calibrated_sample_nums=4096)
test_cfg = dict(type='mmrazor.AutoSlimTestLoop', calibrated_sample_nums=4096)

# auto_scale_lr = dict(base_batch_size=2048)
