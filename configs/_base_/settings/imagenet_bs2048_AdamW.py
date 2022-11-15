# dataset settings
dataset_type = 'mmcls.ImageNet'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = preprocess_cfg['mean'][::-1]
bgr_std = preprocess_cfg['std'][::-1]

# Refers to `_RAND_INCREASING_TRANSFORMS` in pytorch-image-models
rand_increasing_policies = [
    dict(type='mmcls.AutoContrast'),
    dict(type='mmcls.Equalize'),
    dict(type='mmcls.Invert'),
    dict(type='mmcls.Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='mmcls.Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='mmcls.Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='mmcls.SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='mmcls.ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='mmcls.Contrast',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='mmcls.Brightness',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='mmcls.Sharpness',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='mmcls.Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='mmcls.Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
    dict(
        type='mmcls.Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='horizontal'),
    dict(
        type='mmcls.Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='vertical')
]

train_pipeline = [
    dict(type='mmcls.LoadImageFromFile'),
    dict(
        type='mmcls.RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='mmcls.RandAugment',
        policies=rand_increasing_policies,
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='mmcls.RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='mmcls.LoadImageFromFile'),
    dict(
        type='mmcls.ResizeEdge',
        scale=248,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='mmcls.CenterCrop', crop_size=224),
    dict(type='mmcls.PackClsInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='mmcls.RepeatAugSampler'),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=256,
    num_workers=6,
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
paramwise_cfg = dict(
    bias_decay_mult=0.0, norm_decay_mult=0.0, dwconv_decay_mult=0.0)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.002,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }))

# leanring policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        # about 10000 iterations for ImageNet-1k
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=500,
        eta_min=1e-5,
        by_epoch=True,
        begin=20,
        end=500,
        convert_to_iter_based=True),
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=500)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(base_batch_size=2048)
