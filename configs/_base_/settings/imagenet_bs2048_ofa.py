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
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

extra_params = dict(
    translate_const=int(224 * 0.45),
    img_mean=tuple(round(x) for x in data_preprocessor['mean']),
)

train_pipeline = [
    dict(type='mmcls.LoadImageFromFile'),
    dict(type='mmcls.RandomResizedCrop', scale=224),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.ColorJitter', brightness=0.1254, saturation=0.5),
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
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.8, momentum=0.9, weight_decay=0.00001, nesterov=True),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))

# learning policy
max_epochs = 360
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=3125),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='mmrazor.SubnetValLoop', calibrate_sample_num=4096)
test_cfg = dict(type='mmrazor.SubnetValLoop', calibrate_sample_num=4096)
